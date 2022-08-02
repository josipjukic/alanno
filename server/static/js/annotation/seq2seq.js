import Vue from 'vue';
import annotationMixin from './annotations';
import HTTP from '../http';

Vue.use(require('vue-shortkey'));


const vm = new Vue({
  el: '#main',
  delimiters: ['[[', ']]'],
  data: {
    newTodo: '',
    editedTodo: null,
  },
  mixins: [annotationMixin],
  directives: {
    'todo-focus': function (el, binding) {
      if (binding.value) {
        el.focus();
      }
    },
  },

  methods: {
    addTodo() {
      const value = this.newTodo && this.newTodo.trim();
      if (!value) {
        return;
      }

      const docId = this.docs[this.doc_index].id;
      const payload = {
        text: value,
      };
      HTTP.post(`docs/${docId}/annotations/`, payload).then((response) => {
        this.annotations[this.doc_index].push(response.data);
      });

      this.newTodo = '';
    },

    removeTodo(todo) {
      const docId = this.docs[this.doc_index].id;
      HTTP.delete(`docs/${docId}/annotations/${todo.id}`).then((response) => {
        const index = this.annotations[this.doc_index].indexOf(todo);
        this.annotations[this.doc_index].splice(index, 1);
      });
    },

    editTodo(todo) {
      this.beforeEditCache = todo.text;
      this.editedTodo = todo;
    },

    doneEdit(todo) {
      if (!this.editedTodo) {
        return;
      }
      this.editedTodo = null;
      todo.text = todo.text.trim();
      if (!todo.text) {
        this.removeTodo(todo);
      }
      const docId = this.docs[doc_index].id;
      HTTP.put(`docs/${docId}/annotations/${todo.id}`, todo).then((response) => {
        console.log(response);
      });
    },

    cancelEdit(todo) {
      this.editedTodo = null;
      todo.text = this.beforeEditCache;
    },
  },

});
