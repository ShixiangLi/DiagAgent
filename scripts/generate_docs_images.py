"""
Generate topology and diagnostic flow diagrams — light theme, real edges.

Outputs to docs/images/
"""

import json, os, re, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.topology.topology_builder import TopologyBuilder, EDGE_FEEDS, EDGE_HAS_PART
from src.utils.io_utils import load_yaml

OUT_DIR = "docs/images"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "chiller_plant": "#1565C0", "boiler_plant": "#C62828",
    "sdahu": "#2E7D32", "ddahu": "#558B2F",
    "rtu": "#E65100", "fcu": "#6A1B9A",
    "pfpu": "#00695C", "sfpu": "#283593",
}
SYSTEM_LABELS = {
    "chiller_plant": "Chiller Plant", "boiler_plant": "Boiler Plant",
    "sdahu": "Single-Duct AHU", "ddahu": "Dual-Duct AHU",
    "rtu": "Rooftop Unit", "fcu": "Fan Coil Unit",
    "pfpu": "Parallel FPU", "sfpu": "Series FPU",
}
MEDIUM_STYLES = {
    "chilled_water":   ("#1E88E5", "dashed"),
    "hot_water":       ("#E53935", "dotted"),
    "conditioned_air": ("#43A047", "solid"),
}
STATUS_COLORS = {"Normal": "#388E3C", "Abnormal": "#F57C00", "Fault": "#D32F2F"}

def _brick_color(brick):
    b = brick.lower()
    if "coil" in b or "heat" in b: return "#E53935"
    if "fan" in b or "pump" in b: return "#1E88E5"
    if "damper" in b: return "#FB8C00"
    if "zone" in b or "room" in b: return "#43A047"
    if "vav" in b or "terminal" in b or "fpu" in b: return "#8E24AA"
    if "valve" in b: return "#78909C"
    if "chiller" in b: return "#0D47A1"
    if "boiler" in b: return "#B71C1C"
    if "tower" in b or "condenser" in b: return "#00838F"
    if "compressor" in b: return "#E65100"
    return "#455A64"

# ── Infer logical edges for systems without TTL feeds edges ──────────────
INFERRED_EDGES = {
    "chiller_plant": [
        ("Simulated_Chiller_Plant", "Chiller_1"), ("Simulated_Chiller_Plant", "Chiller_2"),
        ("Simulated_Chiller_Plant", "Chiller_3"),
        ("Chiller_1", "Chilled_Water_System"), ("Chiller_2", "Chilled_Water_System"),
        ("Chiller_3", "Chilled_Water_System"),
        ("Primary_Chilled_Water_Loop_Pump_1", "Chilled_Water_System"),
        ("Primary_Chilled_Water_Loop_Pump_2", "Chilled_Water_System"),
        ("Primary_Chilled_Water_Loop_Pump_3", "Chilled_Water_System"),
        ("Secondary_Chilled_Water_Loop_Pump_1", "Chilled_Water_System"),
        ("Secondary_Chilled_Water_Loop_Pump_2", "Chilled_Water_System"),
        ("Chilled_Water_System", "Bypass_Valve"),
        ("Cooling_Tower_1", "Condenser_Water_System"), ("Cooling_Tower_2", "Condenser_Water_System"),
        ("Cooling_Tower_3", "Condenser_Water_System"),
        ("Condenser_Water_System", "Chiller_1"), ("Condenser_Water_System", "Chiller_2"),
        ("Condenser_Water_System", "Chiller_3"),
        ("Condenser_Water_Pump_1", "Condenser_Water_System"),
        ("Condenser_Water_Pump_2", "Condenser_Water_System"),
        ("Condenser_Water_Pump_3", "Condenser_Water_System"),
        ("Condenser_Water_System", "Threeway_Valve"),
    ],
    "boiler_plant": [
        ("Simulated_Boiler_Plant", "Boiler_1"), ("Simulated_Boiler_Plant", "Boiler_2"),
        ("Boiler_1", "HW_Pump_1"), ("Boiler_2", "HW_Pump_2"),
    ],
    "fcu": [],
    "rtu": [("RTU", "compressor"), ("RTU", "condenser"), ("RTU", "supply_fan")],
}

# ================================================================
# 1. Cross-System Overview
# ================================================================
def draw_cross_system(builder, config):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    pos = {
        "chiller_plant": (0.25, 0.82), "boiler_plant": (0.75, 0.82),
        "sdahu": (0.12, 0.48), "ddahu": (0.38, 0.48),
        "rtu": (0.62, 0.48), "fcu": (0.88, 0.48),
        "pfpu": (0.30, 0.15), "sfpu": (0.70, 0.15),
    }

    # Tier backgrounds
    for (y0, y1, label) in [(0.68, 0.96, "Water-Side Plants"),
                             (0.32, 0.64, "Air Handling Units"),
                             (0.02, 0.28, "Terminal Units")]:
        rect = FancyBboxPatch((0.02, y0), 0.96, y1-y0, boxstyle="round,pad=0.015",
                               facecolor="#ECEFF1", edgecolor="#B0BEC5", linewidth=1,
                               transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        ax.text(0.98, y1-0.02, label, ha="right", va="top", fontsize=9,
                color="#78909C", fontstyle="italic", transform=ax.transAxes, zorder=2)

    # Edges
    for link in config.get("cross_system_links", []):
        src, tgt = link["source_system"], link["target_system"]
        medium = link["medium"]
        color, ls = MEDIUM_STYLES.get(medium, ("#888", "solid"))
        x1, y1 = pos[src]; x2, y2 = pos[tgt]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                     alpha=0.6, linestyle=ls,
                                     connectionstyle="arc3,rad=0.08"),
                     transform=ax.transAxes, zorder=3)

    # Nodes
    for sys_id, (x, y) in pos.items():
        comps = builder.get_system_components(sys_id)
        n_c, n_s = len(comps), sum(len(c.get("sensor_names",[])) for c in comps)
        color = COLORS[sys_id]
        r = 0.065 + n_c * 0.0018
        circle = plt.Circle((x, y), r, color=color, alpha=0.9,
                             transform=ax.transAxes, zorder=5, ec="white", lw=2)
        ax.add_patch(circle)
        ax.text(x, y+0.008, SYSTEM_LABELS[sys_id], ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
                transform=ax.transAxes, zorder=6)
        ax.text(x, y-0.018, f"{n_c} comp / {n_s} sens", ha="center", va="center",
                fontsize=6.5, color="#E0E0E0", transform=ax.transAxes, zorder=6)

    # Legend
    handles = [mpatches.Patch(color=c, label=m.replace("_"," ").title())
               for m, (c, _) in MEDIUM_STYLES.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              frameon=True, facecolor="white", edgecolor="#CCC")

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("LBNL Building HVAC — Cross-System Topology",
                 fontsize=15, fontweight="bold", color="#212121", pad=12)
    fig.savefig(os.path.join(OUT_DIR, "topology_overview.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: topology_overview.png")


# ================================================================
# 2. Per-System Topology (with real edges)
# ================================================================
def draw_system_topology(builder, sys_id, config):
    comps = builder.get_system_components(sys_id)
    if not comps: return
    sys_name = SYSTEM_LABELS.get(sys_id, sys_id)
    n_comps = len(comps)

    G = nx.DiGraph()
    comp_map = {}
    for c in comps:
        nid = c["node_id"]
        name = nid.split("::")[-1]
        brick = c.get("brick_class", "Other")
        sensors = c.get("sensor_names", [])
        G.add_node(name, brick=brick, n_sensors=len(sensors), sensor_names=sensors)
        comp_map[nid] = name

    # Real feeds edges from the topology graph
    for u, v, d in builder.graph.edges(data=True):
        if d.get("relation") == EDGE_FEEDS and u in comp_map and v in comp_map:
            G.add_edge(comp_map[u], comp_map[v])

    # Add inferred edges for systems without TTL feeds
    if sys_id in INFERRED_EDGES:
        nodes_set = set(G.nodes())
        for src, tgt in INFERRED_EDGES[sys_id]:
            if src in nodes_set and tgt in nodes_set:
                G.add_edge(src, tgt)

    # Layout
    if G.number_of_edges() > 0:
        # Hierarchical layout using topological sort levels
        try:
            # Compute depth from roots
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if not roots:
                roots = [max(G.nodes(), key=lambda n: G.out_degree(n))]
            depth = {}
            for root in roots:
                for n in nx.bfs_tree(G, root):
                    d = nx.shortest_path_length(G, root, n)
                    depth[n] = max(depth.get(n, 0), d)
            for n in G.nodes():
                if n not in depth:
                    depth[n] = 0

            # Assign positions by depth level
            max_depth = max(depth.values()) if depth else 0
            levels = {}
            for n, d in depth.items():
                levels.setdefault(d, []).append(n)

            pos = {}
            for d, nodes in levels.items():
                nodes.sort()
                n_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    x = (i + 0.5) / max(n_nodes, 1)
                    y = 1.0 - d / max(max_depth, 1)
                    pos[node] = (x, y)
        except:
            pos = nx.spring_layout(G, k=2.0, seed=42)
    else:
        pos = nx.spring_layout(G, k=2.5, seed=42)

    # Draw
    fig_w = max(10, min(16, n_comps * 0.6 + 3))
    fig_h = max(6, min(11, n_comps * 0.4 + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#90A4AE",
                                arrows=True, arrowstyle="-|>", arrowsize=15,
                                width=1.8, alpha=0.7,
                                connectionstyle="arc3,rad=0.05")

    # Nodes
    for node in G.nodes():
        data = G.nodes[node]
        x, y = pos[node]
        color = _brick_color(data["brick"])
        n_s = data["n_sensors"]
        size = 500 + n_s * 100

        nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                node_size=[size], node_color=[color],
                                alpha=0.9, edgecolors="white", linewidths=2)

    # Labels
    labels = {n: n.replace("_", "\n") if len(n) > 15 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6.5,
                             font_color="#212121", font_weight="bold")

    # Sensor count annotations
    for node, (x, y) in pos.items():
        n_s = G.nodes[node]["n_sensors"]
        if n_s > 0:
            ax.text(x, y - 0.08, f"{n_s} sensors", ha="center", fontsize=5.5,
                    color="#757575", fontstyle="italic",
                    transform=ax.transData)

    # Legend
    brick_types = set(G.nodes[n]["brick"] for n in G.nodes())
    seen_colors = {}
    for bt in sorted(brick_types):
        c = _brick_color(bt)
        cat = bt.split("_")[0] if "_" in bt else bt
        if c not in seen_colors:
            seen_colors[c] = bt
    handles = [mpatches.Patch(color=c, label=l) for c, l in seen_colors.items()]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=6.5,
                  frameon=True, facecolor="white", edgecolor="#DDD")

    n_sensors = sum(G.nodes[n]["n_sensors"] for n in G.nodes())
    n_edges = G.number_of_edges()
    ax.set_title(f"{sys_name} ({sys_id}) — {n_comps} Components, {n_sensors} Sensors, {n_edges} Connections",
                 fontsize=13, fontweight="bold", color="#212121", pad=10)
    ax.axis("off")
    fig.savefig(os.path.join(OUT_DIR, f"topology_{sys_id}.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: topology_{sys_id}.png")


# ================================================================
# 3. Diagnostic Flow Diagrams
# ================================================================
def extract_tool_sequence(entry):
    steps = []
    convs = entry["conversations"]
    for i, c in enumerate(convs):
        if c["from"] == "gpt" and "<tool_call>" in c.get("value", ""):
            tm = re.search(r'"name":\s*"(\w+)"', c["value"])
            am = re.search(r'"node_id":\s*"([^"]+)"', c["value"])
            sid = re.search(r'"system_id":\s*"([^"]+)"', c["value"])
            tool_name = tm.group(1) if tm else "?"
            node_id = am.group(1) if am else (sid.group(1) if sid else "")
            result_info = {}
            if i+1 < len(convs) and convs[i+1]["from"] == "observation":
                try:
                    r = json.loads(convs[i+1]["value"])
                    result_info = {
                        "status": r.get("status",""), "fault_type": r.get("fault_type",""),
                        "confidence": r.get("confidence",""),
                        "sensor_values": {k: round(v,2) if isinstance(v,float) else v
                                          for k,v in list(r.get("sensor_readings",{}).items())[:3]},
                        "direction": r.get("suggested_direction",""),
                    }
                except: pass
            steps.append({"tool": tool_name, "node_id": node_id, "result": result_info})
    return steps


def draw_diagnostic_flow(entry, scenario_type, filename):
    meta = entry["metadata"]
    gt = meta["ground_truth"]
    steps = extract_tool_sequence(entry)
    n_steps = len(steps)

    fig_h = max(5, n_steps * 1.1 + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    scenario_id = meta.get("scenario_id", entry["id"])
    fault_type = gt.get("fault_type", "N/A")
    root_node = gt.get("root_cause_node", "N/A")

    ax.text(0.5, 1.0, f"Diagnostic Flow: {scenario_type.replace('_',' ').title()}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold", color="#212121")
    ax.text(0.5, 0.97, f"Scenario: {scenario_id}  |  Fault: {fault_type}  |  Root: {root_node}  |  Steps: {n_steps}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#757575")

    x_tool, x_result = 0.28, 0.70
    y_start = 0.92
    y_step = min(0.075, 0.82 / max(n_steps + 1, 1))

    # User query
    user_q = ""
    for c in entry["conversations"]:
        if c["from"] == "human": user_q = c["value"][:90]; break

    y = y_start
    box = FancyBboxPatch((0.04, y-0.013), 0.92, 0.026, boxstyle="round,pad=0.005",
                          facecolor="#1565C0", transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(0.5, y, f"User: {user_q}", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color="white", zorder=4)
    y -= y_step

    for i, step in enumerate(steps):
        tool, node_id, result = step["tool"], step["node_id"], step["result"]
        short_node = node_id.split("::")[-1] if node_id else ""
        sys_pre = node_id.split("::")[0] if "::" in node_id else ""

        # Tool type styling
        if "diagnose" in tool:
            tc = "#5E35B1"
        elif "upstream" in tool or "downstream" in tool:
            tc = "#00796B"
        else:
            tc = "#546E7A"

        # Tool box
        box = FancyBboxPatch((0.03, y-0.012), 0.44, 0.024, boxstyle="round,pad=0.004",
                              facecolor=tc, alpha=0.85, transform=ax.transAxes, zorder=3)
        ax.add_patch(box)
        ax.text(x_tool, y, f"{tool}({short_node})", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="white", zorder=4)

        # Result box
        status = result.get("status", "")
        if status in STATUS_COLORS:
            sc = STATUS_COLORS[status]
            conf = result.get("confidence", "")
            conf_s = f" {conf:.0%}" if isinstance(conf, (int, float)) else ""
            fault = result.get("fault_type", "")
            direction = result.get("direction", "")
            sensors = result.get("sensor_values", {})

            label_parts = [f"{status}{conf_s}"]
            if fault and fault != "None": label_parts.append(fault)
            if direction: label_parts.append(f">> {direction}")
            label = " | ".join(label_parts)

            s_text = "  ".join(f"{k}={v}" for k,v in sensors.items())

            bh = 0.024
            rbox = FancyBboxPatch((0.50, y-bh/2), 0.47, bh, boxstyle="round,pad=0.004",
                                   facecolor=sc, alpha=0.12, edgecolor=sc, linewidth=1.5,
                                   transform=ax.transAxes, zorder=3)
            ax.add_patch(rbox)
            ax.text(0.52, y+0.003, label, transform=ax.transAxes, ha="left", va="center",
                    fontsize=7, color=sc, fontweight="bold", zorder=4)
            if s_text:
                ax.text(0.52, y-0.008, s_text[:60], transform=ax.transAxes, ha="left",
                        va="center", fontsize=5, color="#616161", fontfamily="monospace", zorder=4)

            ax.annotate("", xy=(0.50, y), xytext=(0.47, y),
                         arrowprops=dict(arrowstyle="->", color=sc, lw=1.2),
                         transform=ax.transAxes)

        # Connector
        if i < n_steps - 1:
            ny = y - y_step
            ax.annotate("", xy=(x_tool, ny+0.015), xytext=(x_tool, y-0.015),
                         arrowprops=dict(arrowstyle="-|>", color="#BDBDBD", lw=0.8),
                         transform=ax.transAxes)
            # Cross-system label
            nn = steps[i+1].get("node_id","")
            ns = nn.split("::")[0] if "::" in nn else ""
            if sys_pre and ns and sys_pre != ns and "diagnose" in steps[i+1]["tool"]:
                ax.text(x_tool+0.26, (y+ny)/2, f"* {sys_pre} -> {ns}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="#F57C00", fontstyle="italic")
        y -= y_step

    # Diagnosis box
    y -= 0.01
    box = FancyBboxPatch((0.12, y-0.018), 0.76, 0.036, boxstyle="round,pad=0.006",
                          facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=2,
                          transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(0.5, y, f"DIAGNOSIS:  {fault_type}  at  {root_node}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#1B5E20", fontweight="bold", zorder=4)

    ax.text(0.98, 0.01, "Data source: Real Oracle Model + LBNL_FDD CSV",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="#9E9E9E")

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ================================================================
# Main
# ================================================================
def main():
    print("Building topology...")
    builder = TopologyBuilder("configs/topology_config.yaml", "data/lbnl")
    builder.build()
    config = load_yaml("configs/topology_config.yaml")

    print("\n1. Cross-System Overview")
    draw_cross_system(builder, config)

    print("\n2. Per-System Topologies")
    for sys_id in config["systems"]:
        draw_system_topology(builder, sys_id, config)

    print("\n3. Diagnostic Flow Diagrams")
    sft_path = "outputs/data/sft_train.jsonl"
    if os.path.exists(sft_path):
        targets = {"single_system": None, "cross_system": None, "no_fault": None}
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                m = e["metadata"]; st = m["scenario_type"]
                if st in targets and targets[st] is None:
                    if st == "single_system" and m.get("path_length",0)>=2 and m["n_tool_calls"]>=6:
                        targets[st] = e
                    elif st == "cross_system" and m.get("path_length",0)>=3:
                        targets[st] = e
                    elif st == "no_fault":
                        targets[st] = e
                if all(v is not None for v in targets.values()): break
        for stype, entry in targets.items():
            if entry:
                draw_diagnostic_flow(entry, stype, f"diag_flow_{stype}.png")

    print(f"\nAll diagrams saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
