# Recursive-Friendly Zero-Knowledge Proof System

## Overview
This project implements a **recursive-friendly zero-knowledge proof system** in Rust using **Arkworks** for finite field arithmetic and **Nalgebra** for matrix operations. The system follows an interactive proof protocol, allowing a prover and verifier to iteratively exchange messages, ensuring the correctness of computations.

## Features
- **Finite Field Operations**: Uses `ark_bn254::Fr` for arithmetic operations in a secure finite field.
- **Matrix & Vector Algebra**: Implements **Kronecker product**, **custom matrix-vector multiplications**, and **inverse operations**.
- **Recursive Proof System**: Iterative **prover-verifier interaction** based on challenge-response rounds.
- **Commitment Scheme**: Implements `commit_section4()` for commitments and `open_poly()` for verification.
- **ZK Proof Verification**: Ensures constraints hold across multiple rounds of interaction.

## Dependencies
- [Arkworks](https://github.com/arkworks-rs) for finite field operations.
- [Nalgebra](https://docs.rs/nalgebra/latest/nalgebra/) for matrix computations.
- [Rand](https://docs.rs/rand/latest/rand/) for random number generation.

## Installation
To run the project, ensure you have Rust installed. You can install Rust using [Rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository:

```sh
git clone https://github.com/your_username/zkp-recursive-proof.git
cd zkp-recursive-proof
```

Install dependencies and build the project:

```sh
cargo build --release
```

Run the main proof system:

```sh
cargo run
```

## Project Structure
```
├── src
│   ├── main.rs             # Entry point of the 
├── Cargo.toml              # Rust package manifest (dependencies)
└── README.md               # Project documentation
```

## Usage
The proof system follows an **interactive protocol** between the prover and verifier. Here’s a high-level workflow:

1. **Commitment Phase**: The prover commits to the initial witness using `commit_section4()`.
2. **Interactive Proof Rounds**:
   - The prover computes proof elements `z_i`.
   - The verifier challenges the prover with `c_i`.
   - The prover updates its state using `next_instance()` and `next_witness()`.
3. **Verification**: The verifier ensures all constraints hold using `open_poly()`.

### Example Execution
Upon running `cargo run`, the output should indicate whether the proof system accepts or rejects:

```
=== Demo: ZK Proof System with Arkworks ===
Generated f (gadget-encoded): [...]
commit => t.len=..., s_arr.len=..., u.len=...
Protocol result => true
```

## References
- [Arkworks Documentation](https://arkworks.rs/)
- [Nalgebra Linear Algebra Library](https://nalgebra.org/)

