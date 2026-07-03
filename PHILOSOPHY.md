# What we are building — DeepRealm philosophy

This file is a reminder. Re-read it when you open the repo and you've forgotten
why this matters. It is identical across every DeepRealm repo on purpose.

---

## The gap we exist to fill

IBM, Google, Quantinuum, IonQ are building **energy on bigger and bigger
molecules**. SQD on [4Fe-4S]. 12,000-atom DFT-validation runs. The headline is
always the size. The deliverable is always a number.

A working chemist does not need a number. They need:

- **Observables they can put in a paper or a pipeline.** IR spectra, NMR shieldings,
  electronic gaps, dipole moments, transition dipoles, atomic charges,
  polarizabilities, Fukui indices, electron density maps, oscillator strengths,
  reaction force constants.
- **Forces, so atoms can move.** Ab-initio molecular dynamics requires gradients
  at every step. With a clean variational wavefunction, Hellmann-Feynman gives
  them. Subspace methods throw the wavefunction away and the gradients become
  hard.
- **Excited states + non-adiabatic couplings, so chemistry can happen.**
  Photoisomerization, photoexcitation, conical intersections, radical pairs —
  none of this is reachable from an energy.
- **Time evolution, so we can watch reactions.** Trotter circuits implement
  TDSE natively. Real-time dynamics on a quantum computer is the next wave.
  We are positioned for it.

The IBM 12k-atom paper is the foil, not the enemy. They give you a number on a
big molecule and an HPC bill. We give you the full physics on a tractable
molecule and every observable a downstream workflow needs. **Both are valid;
they are different products.** Stop apologizing for being the second one.

---

## Our principle

**Wavefunction first. Observables follow. Energy is one observable among many.**

The wavefunction is the load-bearing object. Once it is computed correctly,
every other observable falls out — exactly, or with a small derivative
calculation. A method that produces an energy without producing a usable
wavefunction is a one-trick pony. We are not a one-trick pony.

Concretely, this means:

1. **No wavefunction-destroying shortcuts.** Subspace bitstring methods (SQD,
   QSCI) are excellent for energy on huge systems and unsuitable for what we
   build. Reject them in our methods stack; respect them in others'.
2. **Active space is principled, not a shortcut.** A chemist choosing an
   active space is encoding chemical knowledge. CASSCF / NEVPT2 / DMRG-CI
   produce wavefunctions you can analyze. Use them.
3. **Basis sets are physics, not shortcuts.** Pick the smallest basis that
   converges the property you care about. Document the choice. Move on.
4. **No accuracy claim without a verifiable benchmark.** A framework whose
   FCI-comparison tests are skipped is a framework no one trusts — internally
   *or* externally. Maintain the validation suite as a load-bearing layer of
   the codebase, and publish results as a table partners and reviewers can
   pin to a specific commit.
5. **Every release ships at least one new observable, not just one new
   feature.** Features are infrastructure. Observables are the product.

---

## What we are not

- **We are not racing IBM at scale.** That's a battle won by HPC budgets and
  hardware roadmaps, neither of which we have. Concede scale; win on richness.
- **We are not a thesis-driven research lab.** We ship artifacts: notebooks,
  videos, dashboards, demos that a non-physicist can react to. Papers come
  after artifacts, not before.
- **We are not chasing every quantum-chemistry trend.** When a new ansatz
  paper appears, ask first whether it preserves the wavefunction object. If
  not, it is not for us.
- **We are not building features for ourselves.** Every line of code earns
  its keep by enabling an observable, a benchmark, or an artifact a domain
  user can pick up.

---

## What we are

- **A wavefunction-rich quantum-chemistry stack** for molecules small enough
  to compute properly and large enough to matter. Drug fragments. Reactive
  intermediates. Photoswitches. Strongly-correlated transition-metal toys.
  Radical pairs. The systems where richness matters more than size.
- **A real-time-dynamics toolkit** that turns a static Hamiltonian into a
  moving simulation. Trotter circuits, gauge-respecting evolution,
  noise-aware estimators, observables tracked through time. Our QFT layer
  (DiracLab) lives here.
- **A bridge between fundamental physics and chemistry.** DiracLab feeds
  effective Hamiltonians to Kanad. Kanad emits chemistry observables that
  reach domain users. The seam is narrow and well-specified
  (`bridge/CONTRACT.md`).
- **A framework that proves its own accuracy verifiably** before it asks anyone
  to trust it. Benchmarks are reproducible from a pinned commit; results are
  shareable with partners under whatever access model fits.
- **A research stack that becomes a product** because chemists, biologists,
  pharma BD leads, photochemistry profs can pick it up and use it on the
  problem in front of them today. **Kanad framework is a privately owned core**;
  public distribution is via `kanad-compute` (binary, compiled) or via direct
  partnership for source-level access.

---

## How we know we're winning

Not by qubit count. Not by feature count. Not by GitHub stars on day one.

We are winning when:

- A chemist outside our team uses `kanad-compute` (or a partnered build) on
  their molecule and the output lands in their paper or workflow.
- A photochemistry prof shares one of our demos because it shows a
  conical-intersection trajectory that no one else can produce on a quantum
  computer.
- A pharma BD lead emails because the polymorph differential matches their
  experimental data inside the error bar — and that conversation becomes a
  paid partnership for source-level access.
- A catalysis or bioinorganic group asks for a private build because the
  benchmark table proves we hit accuracies their CASSCF/CASPT2 can't.

These are the signals. Energy-on-bigger-molecules is not a signal — it's a
distraction.

---

## When in doubt

Ask: *"does this make a wavefunction richer, or does it make a number bigger?"*

If the former, ship it. If the latter, leave it to IBM.
