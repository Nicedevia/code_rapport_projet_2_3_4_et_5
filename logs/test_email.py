from email_alert import send_error_email

send_error_email("🚨 Test alerte MCO", "Ceci est un test d'envoi d'alerte par mail.")
def test_force_error(client):
    response = client.get("/force-error")
    assert response.status_code == 500
