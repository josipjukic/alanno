# Generated by Django 3.2.7 on 2022-03-23 13:00

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('server', '0012_document_indexing'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='gl_annotated',
            field=models.BooleanField(default=False, null=True),
        ),
        migrations.AddField(
            model_name='documentannotation',
            name='gl_annotation',
            field=models.BooleanField(default=False, null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='gl_enabled',
            field=models.BooleanField(default=False, null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='main_annotator',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='main_annotator_in_projects', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='sequenceannotation',
            name='gl_annotation',
            field=models.BooleanField(default=False, null=True),
        ),
    ]