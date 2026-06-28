
import os
import pandas as pd
import json
from datetime import datetime

def generate_project_summary(df, metrics_df, output_file="results/project_summary.md"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    summary = f"""# AI Cosmic Intelligence - Project Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview
- **Total planets analyzed**: {len(df)}
- **Potentially habitable (labeled)**: {df['habitability_label'].sum()}
- **Average habitability score**: {df['overall_habitability_score'].mean():.2%}

## Top 5 Habitable Planets
{df.nlargest(5, 'overall_habitability_score')[['planet_name', 'hostname', 'overall_habitability_score', 'prediction_confidence']].to_markdown(index=False)}

## Model Performance Summary
{metrics_df.to_markdown()}
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ Project summary generated at {output_file}")
    return output_file


def save_experiment_log(params, output_file="results/experiment_log.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    log = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params
    }
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            existing_logs = json.load(f)
        existing_logs.append(log)
        logs = existing_logs
    else:
        logs = [log]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, default=str)

    return output_file
