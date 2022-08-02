from abc import ABC, abstractmethod
from random import sample


# class Task(ABC):
#     @abstractmethod
#     def __init__(self, project, language, seed, multilabel=False):
#         self.project_id = project.id
#         self.language = language
#         self.seed = seed
#         self.ids = set()
#         self.selected_ids = set()
#         self.multilabel = multilabel

#     def upload_data(self, documents):
#         new_ids = {doc.id for doc in documents}
#         self.ids.update(new_ids)

#     def clear_ids(self):
#         self.ids.clear()
#         self.selected_ids.clear()
#         print(f"Ids: {self.ids}")
#         print(f"Selected ids: {self.selected_ids}")

#     def remove_id(self, id):
#         self.ids.remove(id)
#         if id in self.selected_ids:
#             self.selected_ids.remove(id)

#     @abstractmethod
#     def select_batch(self, batch_size, **kwargs):
#         pass

#     def get_model_performance(self):
#         return None


# class SimpleAnnoTask(Task):
#     def __init__(self, project, language, seed=42, multilabel=False):
#         super().__init__(project, language, seed, multilabel)

#     def select_batch(self, batch_size, **kwargs):
#         available = self.ids.difference(self.selected_ids)
#         k = min(len(available), batch_size)
#         new_ids = sample(available, k=k)
#         self.selected_ids.update(new_ids)
#         return new_ids
