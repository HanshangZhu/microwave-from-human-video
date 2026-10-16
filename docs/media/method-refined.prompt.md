# Method figure refinement

Created with the built-in image_gen tool from the user's updated four-stage diagram. The original method.png and method.svg are retained. This is a raster design proposal.

Facts checked against README.md, sim/EXPERIMENTS.md sections 1, 7e and 7f, recon/DEMOS.md, and sim/dit_vis.py.

## Initial edit prompt

Use case: infographic-diagram, precise edit of a scientific method figure.
Input image 1 is the edit target, the user's latest four-stage method diagram. Continue refining THIS diagram, preserving its numbered four-stage conceptual organization, pale blue human-data group, pale green digital-twin group, pale amber simulation/learning group, pale slate evaluation group, white background, clean sans-serif typography, and independent dashed G1 side branch. This is a research README/paper figure, not a marketing poster.

Improve the composition substantially: compact landscape ~3:2 aspect ratio, crisp high-resolution typesetting, aligned panels, consistent generous internal padding, reduced dead space, thin precise arrow lines with small arrowheads, no overlapping labels or crossed lines, clear typography hierarchy. No photos, decorative illustrations, shadows, 3D effects, invented logos or invented metrics. Use dark slate text rather than heavy black. Aim for an elegant editable-vector visual style even though output is raster. Keep main labels legible when downscaled. Do not reproduce the screenshot's browser edge or horizontal page divider. Title simply "Method".

LAYOUT:
Top row: two aligned panels, 1 Human data on left and 2 Digital twin on right, with equal top and bottom alignment.
Middle row: wide horizontal panel 3 Simulation and learning, with its FOUR main nodes arranged left to right.
Bottom row: wide panel 4 Evaluation, with kitchen background, closed-loop rollouts and results arranged as a readable flow.
Small separate dashed G1 side-branch box in the gap below the human panel, connected only to Object keyframes, not to the evaluation or learned policy. Keep it compact and clearly secondary.
The main inputs from top panels merge at the Panda solver using tidy orthogonal connections through whitespace.
Avoid the original long looping room-scan arrow: use matching small circular connector labels "A" at the Scans output and the Kitchen 3DGS input, with short local lines and the label "Room scan". This is the SAME room-scan data connection, not a new data source. All main dataflow arrows solid; only the optional G1 branch dashed.

EXACT CONTENT, respecting spelling, capitalization and symbols, no additional claims:
Panel title: "1 · Human data"
Panel subtitle: "3 × 40 s RGB-D takes · fixed RealSense · no markers"
Three nodes in order:
"Human demonstrations" / "Open → pick → place → close"
"Per-frame extraction" / "Door angle · pot pick/place · camera pose"
"Object keyframes" / "Object states and task order"
Connect first -> second -> third. Object keyframes is the output to Panda solver, label edge "Constraints".

Panel title: "2 · Digital twin"
Panel subtitle: "Measured geometry · scanned appearance"
Two input nodes:
"Measurements" / "Tape + calipers"
"Scans" / "RealSense · Pi3X + depth fusion"
Both lead to:
"Digital-twin assets" / "Microwave URDF + pot mesh · baked colours"
Small specific note inside panel: "Microwave registration RMSE: 2.0–2.2 mm"
Assets output to Panda solver labeled "Scene". Scans also has connector A labeled "Room scan".

Small dashed side branch:
"Unitree G1 upper body"
"Same keyframes · scripted IK"
"Qualitative demo only"
Only receives dashed arrow from Object keyframes; no outgoing arrow.

Panel title: "3 · Simulation and learning"
Panel subtitle: "ManiSkill3 / SAPIEN · PhysX CPU · 20 Hz joint control"
Four horizontally arranged nodes linked left to right:
"Panda solver" / "Cartesian IK in physics"
"96 successful demos" / "58 clean + 38 DART"
"Randomised renders" / "2 scenes per demo"
"Visuomotor policy" / "ResNet-18 + DiT" / "Front + wrist RGB + proprioception"
Policy output connects down to closed-loop Panda rollouts in panel 4, labeled "Action chunks".

Panel title: "4 · Evaluation"
Panel subtitle: "Same 128 unseen starts · paired evaluation"
Left background input, with incoming connector A:
"Kitchen 3DGS"
"300 frames · microwave inpainted out"
"Both camera backgrounds · evaluation only"
Connect background input to middle node:
"Closed-loop Panda rollouts"
"Policy ↔ simulated environment"
This middle node also receives the learned policy output from stage 3.
Connect middle node to a right results node:
"Task success"
"Grey floor 74% · splat room 68%"
Panel footnote:
"v6-long · corrected white-rim pot · simulation only"
Ensure the kitchen NEVER points to training and G1 NEVER points to evaluation.
The typography and graph must be fully accurate. The numbers 74% and 68% are prominently readable but subordinate to the method narrative. Preserve the original scientific meaning; this is polish and clarification, not an expanded claimed system.

## Connector correction prompt

Precisely edit the attached method figure. Keep EVERYTHING unchanged: all text verbatim, all panels and their positions, colours, all nodes, all font sizes, G1 branch, evaluation, other arrows and margins. Only correct the TWO connectors from Object keyframes and Digital-twin assets to the Panda solver in stage 3.

CRITICAL semantic correction: Digital-twin assets must feed Panda solver, NOT "96 successful demos". Remove the current Scene line and arrowhead entering 96 successful demos entirely. There must be NO downward arrow entering 96 successful demos; it must receive only the left-to-right arrow from Panda solver.

Replace the constraints and scene paths with a clean merged connection:
- The Constraints path starts from the RIGHT edge of Object keyframes and bends down into the white gap between the top panels and stage 3.
- The Scene path starts from the BOTTOM edge of Digital-twin assets and also descends into that white gap.
- Join these two paths at a small filled junction dot in that white gap, approximately x=700,y=520 on the 1536x1024 canvas.
- From the junction, ONE shared solid connector goes vertically down at x=700 (well to the RIGHT of all stage-3 header and subtitle text), to y=645, then left horizontally to x=375, then down into the TOP edge of the Panda solver node. Put its only arrowhead at Panda solver.
- Keep the labels "Constraints" and "Scene" next to their respective upstream branches.
- Remove the old Constraints path at x=479 that crosses the stage-3 subtitle, so NO line overlaps ANY word.
- The shared connection means the solver uses both object constraints and the digital-twin scene.
Do not add text. Do not move, redraw or modify any other part of the diagram. Ensure the header "3 · Simulation and learning" and its subtitle are fully unobstructed, and every letter and number remains unchanged.

