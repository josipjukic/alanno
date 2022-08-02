from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegisterForm
from django.urls import reverse_lazy
from django.views.generic import FormView
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth.models import Group
from server.models import UserOptions


class RegisterView(FormView):
    form_class = UserRegisterForm
    success_url = reverse_lazy("login")

    def form_valid(self, form):
        user = form.save()
        anno_group = Group.objects.get(name="annotator")
        anno_group.user_set.add(user)
        user_options = UserOptions(user=user)
        user_options.save()
        messages.success(
            self.request, "Registration was successful. Please log in to continue."
        )
        return HttpResponseRedirect(RegisterView.success_url)
