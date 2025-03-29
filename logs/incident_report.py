import re
import datetime
import os
from email_alert import send_error_email


def generate_incident_report(log_file="logs/app_error.log", report_file="logs/incident_report.md", error_threshold=1):
    if not os.path.exists(log_file):
        print(f"❌ Le fichier {log_file} n'existe pas.")
        return

    errors = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "ERROR" in line:
                errors.append(line.strip())

    if len(errors) >= error_threshold:
        new_content = "# Rapport d'Incident\n\n"
        new_content += f"**Date** : {datetime.datetime.now().isoformat()}\n\n"
        new_content += f"**Nombre d'erreurs détectées** : {len(errors)}\n\n"
        new_content += "## Détails des erreurs\n"
        for err in errors:
            new_content += f"- {err}\n"

        # Vérifier s'il y a un ancien rapport identique
        previous = ""
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                previous = f.read()

        if new_content != previous:
            with open(report_file, "w", encoding="utf-8") as report:
                report.write(new_content)
            print(f"✅ Rapport mis à jour : {report_file}")
            send_error_email(subject="🚨 Nouvelle erreur détectée dans l'API", message=new_content)
        else:
            print("ℹ️ Aucun changement dans le rapport d'erreur.")
    else:
        print("ℹ️ Aucune erreur significative détectée.")


if __name__ == "__main__":
    generate_incident_report()
