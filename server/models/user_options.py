from django.db import models
from django.contrib.auth.models import User


class UserOptions(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, primary_key=True, related_name="options"
    )
    use_color = models.BooleanField(default=False)
