from django import forms
from .models import Label, Project
from django.contrib.auth.models import User
from al.models import TORCH_MODELS, TRANSFORMER_MODELS, RECURSIVE_MODELS


class ProjectForm(forms.ModelForm):
    def __init__(self, user, *args, **kwargs):
        super(ProjectForm, self).__init__(*args, **kwargs)
        self.label_suffix = ""
        self.fields["language"].initial = Project.EN
        self.fields["language"].choices = self.fields["language"].choices[1:]
        self.fields["al_method"].initial = Project.LEAST_CONFIDENT
        self.fields["al_method"].choices = self.fields["al_method"].choices[1:]
        self.fields["model_name"].initial = Project.LOG_REG
        self.fields["model_name"].choices = self.fields["model_name"].choices[1:]
        self.fields["vectorizer_name"].initial = Project.COUNT
        self.fields["vectorizer_name"].choices = self.fields["vectorizer_name"].choices[
            1:
        ]
        self.fields["token_type"].initial = Project.WORDS
        self.fields["token_type"].choices = self.fields["token_type"].choices[1:]
        self.fields["min_ngram"].initial = 1
        self.fields["max_ngram"].initial = 1
        choices = [("", "---------")]
        for choice in Project.PROJECT_CHOICES:
            name = Project.TYPE_MAPPING[choice[0]]
            choice_label = name.lower()
            perm = f"server.add_{choice_label}"
            if user.has_perm(perm):
                choices.append(choice)

        self.fields["project_type"] = forms.ChoiceField(choices=choices)

    def clean(self):
        data = super().clean()
        project_type = data["project_type"]
        al_mode = data["al_mode"]
        al_method = data["al_method"]
        multilabel = data["multilabel"]
        model_name = data["model_name"]
        vectorizer_name = data["vectorizer_name"]
        min_ngram = data["min_ngram"]
        max_ngram = data["max_ngram"]

        if al_mode and project_type not in [Project.DOCUMENT_CLASSIFICATION]:
            raise forms.ValidationError(
                f"{project_type} is not supported in Active Learning mode (yet)."
            )

        if al_method == Project.MULTILABEL_UNCERTAINTY and not multilabel:
            raise forms.ValidationError(
                f"Multi-label uncertainty method can be used only in Multi-label classification projects."
            )

        if (
            al_method in [Project.CORE_SET, Project.BADGE]
            and model_name not in TORCH_MODELS
        ):
            readable_al_method = get_readable_name(al_method, Project.AL_METHOD_CHOICES)
            raise forms.ValidationError(
                f"{readable_al_method} AL method can be used only with Deep Learning models."
            )

        if model_name in RECURSIVE_MODELS and vectorizer_name not in [
            Project.EMB_MATRX
        ]:
            readable_model_name = get_readable_name(
                model_name, Project.MODEL_NAME_CHOICES
            )
            readable_vectorizer_name = get_readable_name(
                vectorizer_name, Project.VECTORIZER_NAME_CHOICES
            )
            raise forms.ValidationError(
                f"{readable_model_name} model does not work with {readable_vectorizer_name} vectorizer."
            )

        if model_name in [Project.MLP] and vectorizer_name in [Project.EMB_MATRX]:
            readable_model_name = get_readable_name(
                model_name, Project.MODEL_NAME_CHOICES
            )
            readable_vectorizer_name = get_readable_name(
                vectorizer_name, Project.VECTORIZER_NAME_CHOICES
            )
            raise forms.ValidationError(
                f"{readable_model_name} model does not work with {readable_vectorizer_name} vectorizer."
            )

        if model_name not in TORCH_MODELS and vectorizer_name in [Project.EMB_MATRX]:
            readable_model_name = get_readable_name(
                model_name, Project.MODEL_NAME_CHOICES
            )
            readable_vectorizer_name = get_readable_name(
                vectorizer_name, Project.VECTORIZER_NAME_CHOICES
            )
            raise forms.ValidationError(
                f"{readable_model_name} model does not work with {readable_vectorizer_name} vectorizer."
            )

        if min_ngram is None or max_ngram is None or min_ngram > max_ngram:
            raise forms.ValidationError(
                "Value of field min ngram can not be bigger then max ngram."
            )

        if min_ngram < 1 or max_ngram < 1:
            raise forms.ValidationError(
                "Values of fields min ngram and max ngram have to be positive integers."
            )

    def clean_vocab_max_size(self):
        data = self.cleaned_data["vocab_max_size"]
        return data * 1000

    class Meta:
        model = Project
        fields = (
            "name",
            "description",
            "project_type",
            "language",
            "multilabel",
            "hierarchy",
            "al_mode",
            "al_method",
            "model_name",
            "vectorizer_name",
            "adjustable_vocab",
            "vocab_max_size",
            "vocab_min_freq",
            "token_type",
            "min_ngram",
            "max_ngram",
        )
        widgets = {
            "multilabel": forms.CheckboxInput(),
            "hierarchy": forms.CheckboxInput(),
            "al_mode": forms.CheckboxInput(),
            "adjustable_vocab": forms.CheckboxInput(),
        }


def get_readable_name(data, choices):
    for choice in choices:
        if choice[0] == data:
            return choice[1]


# class PlainNewBatchForm(forms.Form):
#     batch_size = forms.IntegerField(min_value=1, max_value=100, initial=10)


# class NewBatchForm(forms.Form):
#     batch_size = forms.IntegerField(min_value=1, max_value=100, initial=10)
#     test_proportion = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.2)


# class IRForm(forms.Form):
#     def __init__(self, labels, fixed, *args, **kwargs):
#         super(IRForm, self).__init__(*args, **kwargs)
#         self.fixed = fixed
#         if not fixed:
#             for lab in labels:
#                 self.fields[f"{lab.text}_keywords"] = forms.CharField()
#             self.fields["per_class"] = forms.IntegerField(
#                 min_value=1, max_value=5, initial=1
#             )
#             print(labels)
#             print(labels.count())
