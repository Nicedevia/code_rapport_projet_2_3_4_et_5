import re
import datetime
import sys

def generate_incident_report(log_file, report_file, error_threshold=1):
    errors = []
    # On considère comme erreur toute ligne contenant "ERROR" ou "Exception"
    error_pattern = re.compile(r'ERROR|Exception', re.IGNORECASE)
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if error_pattern.search(line):
                    errors.append(line.strip())
    except FileNotFoundError:
        print(f"Le fichier de log {log_file} n'existe pas.", file=sys.stderr)
        return
    
    if len(errors) >= error_threshold:
        with open(report_file, 'w') as report:
            report.write("# Rapport d'Incident\n\n")
            report.write("**Date** : " + datetime.datetime.now().isoformat() + "\n\n")
            report.write("**Nombre d'erreurs détectées** : " + str(len(errors)) + "\n\n")
            report.write("## Détails des erreurs\n")
            for err in errors:
                report.write("- " + err + "\n")
        print("Rapport d'incident généré.")
    else:
        print("Aucune erreur significative détectée.")

if __name__ == '__main__':
    # Les chemins peuvent être adaptés selon ton projet
    generate_incident_report("application.log", "incident_report.md", error_threshold=1)
