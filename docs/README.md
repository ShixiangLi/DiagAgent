# DiagAgent Documentation

## Architecture & Topology

- [**System Topology Diagrams**](topology_diagrams.md) — Visual maps of all 8 HVAC subsystems and cross-system connections
- [**Diagnostic Flow Examples**](diagnostic_flow_examples.md) — Step-by-step visualization of real SFT diagnostic trajectories

## Pipeline Overview

```mermaid
graph LR
    subgraph DataPrep["1️⃣ Data Preparation"]
        TTL["Brick TTL<br/>Ontology Files"]
        CSV["LBNL_FDD<br/>Sensor CSVs"]
        TOPO["Topology<br/>Builder"]
    end
    
    subgraph Models["2️⃣ Oracle Training"]
        FE["Feature<br/>Engineering"]
        LGB["LightGBM<br/>Oracle Models"]
        REG["Model<br/>Registry"]
    end
    
    subgraph SFT["3️⃣ SFT Data Generation"]
        SCEN["Fault Scenario<br/>Generator"]
        PATH["Diagnostic Path<br/>Generator"]
        TRAJ["Trajectory<br/>Generator"]
        FMT["SFT<br/>Formatter"]
    end
    
    subgraph Training["4️⃣ Agent Training"]
        SFTM["SFT Training<br/><i>Tool-use learning</i>"]
        RL["RL Optimization<br/><i>Search efficiency</i>"]
    end
    
    TTL --> TOPO
    CSV --> FE --> LGB --> REG
    TOPO --> SCEN
    REG --> TRAJ
    SCEN --> PATH --> TRAJ --> FMT
    FMT --> SFTM --> RL
    
    style DataPrep fill:#1e3a5f,color:#fff
    style Models fill:#4a148c,color:#fff
    style SFT fill:#1b5e20,color:#fff
    style Training fill:#b71c1c,color:#fff
```

## Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| Oracle Architecture | LightGBM | Superior on tabular HVAC data vs neural networks |
| Path Routing | Diagnostic Path + Path-Aware Oracle | Ensures logical Normal → Abnormal → Fault progression |
| Sensor Data | Real CSV readings | Agent learns authentic sensor patterns, not synthesized |
| Complexity Control | 3-15 tool calls per trajectory | Easy/Normal/Hard tiers for curriculum learning |
| Cross-System Links | 10 links across 3 media types | Enables multi-hop fault tracing through water/air/thermal |
