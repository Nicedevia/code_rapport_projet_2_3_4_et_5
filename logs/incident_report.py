import re
import datetime
import sys
import os

def generate_incident_report(log_file, report_file, error_threshold=1):
    errors = []
    error_pattern = re.compile(r'ERROR|Exception', re.IGNORECASE)

    if not os.path.exists(log_file):
        print(f"⚠️ Le fichier de log {log_file} n'existe pas.", file=sys.stderr)
        return

    with open(log_file, 'r') as f:
        for line in f:
            if error_pattern.search(line):
                errors.append(line.strip())

    if len(errors) >= error_threshold:
        with open(report_file, 'w') as report:
            report.write("# Rapport d'Incident\n\n")
            report.write("**Date** : " + datetime.datetime.now().isoformat() + "\n\n")
            report.write("**Nombre d'erreurs détectées** : " + str(len(errors)) + "\n\n")
            report.write("## Détails des erreurs\n")
            for err in errors:
                report.write("- " + err + "\n")
        print("✅ Rapport d'incident généré.")
    else:
        print("ℹ️ Aucune erreur significative détectée.")

if __name__ == '__main__':
    # Assure-toi que ces chemins sont corrects dans Docker
    generate_incident_report("logs/app.log", "logs/incident_report.md", error_threshold=1)
