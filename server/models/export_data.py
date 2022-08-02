from django.db import models
from .project import Project
from datetime import datetime
from django.contrib.auth.models import User


class ExportData(models.Model):
    FORMAT_CHOICES = (
        ("csv", "csv"),
        ("json", "json"),
    )

    text = models.TextField(default="")
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="export")
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name="export"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    format = models.CharField(max_length=4, null=True, choices=FORMAT_CHOICES)
    is_aggregated = models.BooleanField(default=None, null=True)
    is_unlabeled = models.BooleanField(default=None, null=True)

    class Meta:
        unique_together = ("user", "project")

    def get_message(self):
        msg = self.format.upper() + " file created "
        now = datetime.now()
        time_passed = now - self.created_at

        minutes_passed = round(time_passed.total_seconds() / 60.0)
        hours_passed = round(time_passed.total_seconds() / (60.0 * 60.0))
        days_passed = round(time_passed.total_seconds() / (60.0 * 60.0 * 24.0))
        if minutes_passed < 60:
            if minutes_passed == 0:
                msg += "less than a minute ago"
            elif minutes_passed == 1:
                msg += "1 minute ago."
            else:
                msg += str(minutes_passed) + " minutes ago."
        elif hours_passed < 24:
            if hours_passed == 1:
                msg += "1 hour ago."
            else:
                msg += str(hours_passed) + " hours ago."

        else:
            if days_passed == 1:
                msg += "1 day ago."
            else:
                msg += str(days_passed) + " days ago."
        return msg
