import re
import datetime
import sys
import os

def generate_incident_report(log_file="logs/app.log", report_file="logs/incident_report.md", error_threshold=1):
    if not os.path.exists(log_file):
        print(f"❌ Le fichier {log_file} n'existe pas.")
        return

    errors = []
    error_pattern = re.compile(r'ERROR.*?(/[\w/-]+).*?(\d{3})', re.IGNORECASE)

    with open(log_file, 'r') as f:
        for line in f:
            if "ERROR" in line:
                errors.append(line.strip())

    if len(errors) >= error_threshold:
        with open(report_file, 'w') as report:
            report.write("# Rapport d'Incident\n\n")
            report.write(f"**Date** : {datetime.datetime.now().isoformat()}\n\n")
            report.write(f"**Nombre d'erreurs détectées** : {len(errors)}\n\n")
            report.write("## Détails des erreurs\n")
            for err in errors:
                report.write(f"- {err}\n")
        print(f"✅ Rapport d'incident généré : {report_file}")
    else:
        print("ℹ️ Aucune erreur significative détectée.")

if __name__ == "__main__":
    generate_incident_report()
