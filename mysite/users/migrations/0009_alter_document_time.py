# Generated by Django 3.2.6 on 2021-10-08 18:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0008_alter_document_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='time',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
