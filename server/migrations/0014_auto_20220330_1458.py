# Generated by Django 3.2.7 on 2022-03-30 14:58

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('server', '0013_auto_20220323_1300'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='documentannotation',
            unique_together={('document', 'user', 'label', 'gl_annotation')},
        ),
        migrations.AlterUniqueTogether(
            name='sequenceannotation',
            unique_together={('document', 'user', 'label', 'start_offset', 'end_offset', 'gl_annotation')},
        ),
    ]
