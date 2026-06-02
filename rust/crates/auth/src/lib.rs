//! RFC 8628 device authorization grant client.
//!
//! Mirrors the `login` Click command in `kanad_compute/cli.py`:
//!   1. `POST /api/auth/device/code` → `device_code`, `user_code`,
//!      `verification_uri`, `interval`, `expires_in`.
//!   2. Display the `user_code` + `verification_uri_complete`, optionally
//!      open the browser.
//!   3. Poll `POST /api/auth/device/token` honouring `interval`, bumping
//!      on `slow_down`, until APPROVED → returns the JWT access token.
//!
//! Refresh-token rotation is intentionally not implemented yet (Phase 2
//! deferred item — 30-day access tokens cover the migration window).

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid url: {0}")]
    Url(#[from] url::ParseError),
    #[error("device authorization denied")]
    AccessDenied,
    #[error("device code expired before approval")]
    Expired,
    #[error("device code invalid or already redeemed")]
    InvalidGrant,
    #[error("polling timed out after {0:?}")]
    Timeout(Duration),
    #[error("unexpected server response: {0}")]
    Unexpected(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    pub interval: u64,
    pub expires_in: u64,
}

#[derive(Debug, Clone)]
pub struct AccessToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub scope: Option<String>,
    pub client_id: Option<String>,
}

#[derive(Deserialize)]
struct TokenSuccess {
    access_token: String,
    #[serde(default = "bearer")]
    token_type: String,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    client_id: Option<String>,
}
fn bearer() -> String {
    "Bearer".into()
}

#[derive(Deserialize)]
struct TokenError {
    error: String,
    #[serde(default)]
    #[allow(dead_code)]
    error_description: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeRequest<'a> {
    pub client_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<&'a str>,
}

pub struct DeviceFlow {
    pub base_url: String,
    pub client_id: String,
    pub scope: Option<String>,
    pub http: reqwest::Client,
}

impl DeviceFlow {
    pub fn new(base_url: impl Into<String>, client_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client_id: client_id.into(),
            scope: None,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    pub async fn request_code(&self) -> Result<DeviceCodeResponse, AuthError> {
        let url = format!("{}/api/auth/device/code", self.base_url);
        let body = CodeRequest {
            client_id: &self.client_id,
            scope: self.scope.as_deref(),
        };
        let r = self.http.post(&url).json(&body).send().await?;
        if !r.status().is_success() {
            return Err(AuthError::Unexpected(format!(
                "device/code returned {}",
                r.status()
            )));
        }
        Ok(r.json::<DeviceCodeResponse>().await?)
    }

    /// Poll the token endpoint until the user approves or the device code
    /// expires. Returns the access token on success.
    pub async fn poll_token(&self, code: &DeviceCodeResponse) -> Result<AccessToken, AuthError> {
        let url = format!("{}/api/auth/device/token", self.base_url);
        let deadline = Instant::now() + Duration::from_secs(code.expires_in);
        let mut interval = Duration::from_secs(code.interval.max(1));

        loop {
            if Instant::now() >= deadline {
                return Err(AuthError::Timeout(Duration::from_secs(code.expires_in)));
            }
            tokio::time::sleep(interval).await;

            let resp = self
                .http
                .post(&url)
                .json(&serde_json::json!({
                    "client_id": self.client_id,
                    "device_code": code.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }))
                .send()
                .await?;

            let status = resp.status();
            let bytes = resp.bytes().await?;

            if status.is_success() {
                let ok: TokenSuccess = serde_json::from_slice(&bytes)
                    .map_err(|e| AuthError::Unexpected(format!("decode token success: {e}")))?;
                return Ok(AccessToken {
                    access_token: ok.access_token,
                    token_type: ok.token_type,
                    expires_in: ok.expires_in,
                    scope: ok.scope,
                    client_id: ok.client_id,
                });
            }

            // Non-2xx → expect an RFC 8628 error body.
            let err: TokenError = serde_json::from_slice(&bytes).map_err(|e| {
                AuthError::Unexpected(format!("decode token error ({status}): {e}"))
            })?;

            match err.error.as_str() {
                "authorization_pending" => {
                    // keep current interval, keep polling
                }
                "slow_down" => {
                    interval += Duration::from_secs(5);
                }
                "access_denied" => return Err(AuthError::AccessDenied),
                "expired_token" => return Err(AuthError::Expired),
                "invalid_grant" => return Err(AuthError::InvalidGrant),
                other => {
                    return Err(AuthError::Unexpected(format!(
                        "unknown error code: {other}"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_code_response() {
        let raw = r#"{
          "device_code":"abc","user_code":"WXYZ-1234",
          "verification_uri":"https://example.com/connect-device",
          "verification_uri_complete":"https://example.com/connect-device?code=WXYZ-1234",
          "interval":5,"expires_in":900
        }"#;
        let r: DeviceCodeResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(r.user_code, "WXYZ-1234");
        assert_eq!(r.interval, 5);
    }

    #[test]
    fn parses_token_success() {
        let raw = r#"{"access_token":"jwt.x.y","token_type":"Bearer","expires_in":2592000,"scope":"compute","client_id":"cli"}"#;
        let r: TokenSuccess = serde_json::from_str(raw).unwrap();
        assert_eq!(r.access_token, "jwt.x.y");
        assert_eq!(r.expires_in, Some(2592000));
    }

    #[test]
    fn parses_token_error_body() {
        let raw = r#"{"error":"authorization_pending","error_description":"not yet"}"#;
        let r: TokenError = serde_json::from_str(raw).unwrap();
        assert_eq!(r.error, "authorization_pending");
    }
}
