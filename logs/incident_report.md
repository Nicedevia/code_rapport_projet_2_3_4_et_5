# Rapport d'Incident

**Date** : 2025-03-27T10:37:00.242479

**Nombre d'erreurs détectées** : 46

## Détails des erreurs
- 2025-03-27 09:50:42,208 - INFO - GET /force-error -> 404
- 2025-03-27 09:50:45,064 - INFO - GET /force-error -> 404
- 2025-03-27 09:51:58,590 - INFO - GET /force-error -> 404
- 2025-03-27 09:59:55,980 - ERROR - Erreur GET /force-error - Erreur volontaire pour test MCO
- During handling of the above exception, another exception occurred:
- await self.app(scope, receive_or_disconnect, send_no_error)
- File "/usr/local/lib/python3.9/site-packages/starlette/middleware/exceptions.py", line 62, in __call__
- await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
- await wrap_app_handling_exceptions(app, request)(scope, receive, send)
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
- File "/app/api/routes.py", line 165, in trigger_error
- raise ValueError("Erreur volontaire pour test MCO")
- ValueError: Erreur volontaire pour test MCO
- 2025-03-27 09:59:57,257 - ERROR - Erreur GET /force-error - Erreur volontaire pour test MCO
- During handling of the above exception, another exception occurred:
- await self.app(scope, receive_or_disconnect, send_no_error)
- File "/usr/local/lib/python3.9/site-packages/starlette/middleware/exceptions.py", line 62, in __call__
- await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
- await wrap_app_handling_exceptions(app, request)(scope, receive, send)
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
- File "/usr/local/lib/python3.9/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
- File "/app/api/routes.py", line 165, in trigger_error
- raise ValueError("Erreur volontaire pour test MCO")
- ValueError: Erreur volontaire pour test MCO
- 2025-03-27 10:21:21,592 - INFO - GET /force-error -> 200
- 2025-03-27 10:23:51,378 - INFO - GET /force-error -> 200
- 2025-03-27 10:24:38,992 - INFO - GET /force-error -> 200
- 2025-03-27 10:24:39,611 - INFO - GET /force-error -> 200
- 2025-03-27 10:24:40,223 - INFO - GET /force-error -> 200
- 2025-03-27 10:26:25,700 - INFO - GET /force-error -> 200
- 2025-03-27 10:26:26,289 - INFO - GET /force-error -> 200
- 2025-03-27 10:26:50,352 - INFO - GET /force-error -> 200
- 2025-03-27 10:28:49,203 - INFO - GET /force-error -> 200
- 2025-03-27 10:28:49,969 - INFO - GET /force-error -> 200
- 2025-03-27 10:28:50,459 - INFO - GET /force-error -> 200
- 2025-03-27 10:28:50,729 - INFO - GET /force-error -> 200
- 2025-03-27 10:28:51,001 - INFO - GET /force-error -> 200
- 2025-03-27 10:33:30,871 - INFO - GET /force-error -> 200
- 2025-03-27 10:33:31,535 - INFO - GET /force-error -> 200
- 2025-03-27 10:34:38,747 - INFO - GET /force-error -> 200
- 2025-03-27 10:34:39,290 - INFO - GET /force-error -> 200
