import hopsworks

# Das öffnet ein Browser-Fenster zum Login oder fragt nach dem API Key
project = hopsworks.login()

# Zugriff auf den Feature Store (Herzstück deines Proposals)
fs = project.get_feature_store()

print(f"Erfolgreich mit Hopsworks-Projekt '{project.name}' verbunden!")