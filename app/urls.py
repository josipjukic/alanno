"""
app URL configuration
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.contrib.auth.views import LoginView, PasswordResetView, LogoutView
from server.urls import router
from .views import RegisterView
from .forms import UserLoginForm


urlpatterns = [
    re_path(r"^celery-progress/", include("celery_progress.urls")),
    path("", include("server.urls")),
    path("admin/", include("loginas.urls")),
    path("admin/", admin.site.urls),
    path(
        "login/",
        LoginView.as_view(
            template_name="general/login.html",
            redirect_authenticated_user=True,
            authentication_form=UserLoginForm,
        ),
        name="login",
    ),
    path("logout/", LogoutView.as_view(), name="logout"),
    path(
        "register/",
        RegisterView.as_view(template_name="general/register.html"),
        name="register",
    ),
    path("password_reset/", PasswordResetView.as_view(), name="password_reset"),
    path("api-auth/", include("rest_framework.urls")),
    path("api/", include(router.urls)),
]
