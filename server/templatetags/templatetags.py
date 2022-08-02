from django import template

register = template.Library()

@register.filter(name="mul")
def mul(value, arg):
    return int(value) * arg if value else None