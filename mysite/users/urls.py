from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static


app_name = "user"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("signup/", views.signup_view, name="signup"),
    path("buttons/", views.buttons_view, name="buttons"),
    path("button1/", views.button1_view, name="button1"),
    path("button2/", views.button2_view, name="button2"),
    path("feedback/", views.feedback_view, name="feedback"),
    path("uploadFile/", views.uploadFile, name="uploadFile"),
    path("uploadFile/time/filecomplete/",views.file_complete,name="filecomplete"),
    path("uploadFile/time/",views.time,name="time"),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )
