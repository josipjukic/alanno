# Generated by Django 3.2.7 on 2022-03-01 12:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0009_alter_userlog_metadata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='al_method',
            field=models.CharField(choices=[('least_conf', 'Least confident'), ('margin', 'Margin'), ('entropy', 'Entropy'), ('entropy_density', 'Entropy + Informative density'), ('multilab_uncert', 'Multi-label uncertainty'), ('core_set', 'Core-set'), ('badge', 'BADGE')], max_length=30, null=True),
        ),
    ]
