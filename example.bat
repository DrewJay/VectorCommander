py src/analysis.py --vector_transition --column "Finding Labels" --label "Hernia" --neutral_label "No Finding" --f_target 2 --f_steps 10 --samples 5
py src/analysis.py --vector_lookup --column "Finding Labels" --label "Hernia" --neutral_label "No Finding"
py src/analysis.py --vector_reconstruction --samples=5
