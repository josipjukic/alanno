import os

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        if User.objects.count() == 0:
            username = os.environ.get("DJANGO_ADMIN_USERNAME") or "admin"
            email = os.environ.get("DJANGO_ADMIN_EMAIL") or "admin@gmail.com"
            password = os.environ.get("DJANGO_ADMIN_PASSWORD") or "morskipassword"
            print(f"Creating account for {username} {email}")
            admin = User.objects.create_superuser(
                email=email, username=username, password=password
            )
            admin.is_active = True
            admin.is_admin = True
            admin.save()
        else:
            print("Admin account already exists. Skipping initialization...")
