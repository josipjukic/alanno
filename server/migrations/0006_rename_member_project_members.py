# Generated by Django 3.2.7 on 2022-01-13 12:26

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0005_project_member'),
    ]

    operations = [
        migrations.RenameField(
            model_name='project',
            old_name='member',
            new_name='members',
        ),
    ]
