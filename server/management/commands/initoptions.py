from server.models import UserOptions
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        for user in User.objects.all():
            if not hasattr(user, "options"):
                print(f"Adding options for user '{user}'")
                o = UserOptions(user=user)
                o.save()

