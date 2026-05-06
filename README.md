# ApexFlow

**Physics-informed neural surrogate for F1 wing aerodynamics, with LLM-orchestrated CFD data generation.**

In Formula 1, a slipstream lets a following car gain speed at a fraction of the energy cost of the leader. ApexFlow does the same for aerodynamic design: a physics-informed neural network rides in the wake of expensive CFD simulations, predicting lift and drag coefficients in microseconds instead of minutes.

## Status

🚧 In active development

## Architecture

Three loosely coupled components:

1. **MCP Server** (`mcp_server/`) — exposes OpenFOAM tools to Claude Desktop, which orchestrates simulation campaigns autonomously across a Latin Hypercube–sampled design space.
2. **Surrogate Model** (`surrogate/`) — physics-informed neural network trained on CFD-generated data, with loss terms enforcing thin-airfoil theory, the drag polar, and symmetry.
3. **Evaluation** (`notebooks/`) — quantifies prediction accuracy and the speedup over direct CFD.

## Tech Stack

- **CFD:** OpenFOAM v12 (Foundation)
- **Agent:** MCP server + Claude Desktop
- **ML:** PyTorch
- **DoE:** Latin Hypercube Sampling (scipy.stats.qmc)
