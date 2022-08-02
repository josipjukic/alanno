import Vue from 'vue';
import annotationMixin from './annotations';
import HTTP from '../http';

Vue.use(require('vue-shortkey'), {
  prevent: ['input', 'textarea'],
});


const vm = new Vue({
  el: '#main',
  delimiters: ['[[', ']]'],
  mixins: [annotationMixin],

  methods: {
    isIn(label) {
      for (let i = 0; i < this.docs[this.doc_index].annotations.length; i++) {
        const a = this.docs[this.doc_index].annotations[i];
        if (a.label === label.id) {
          return a;
        }
      }
      return false;
    },

    async addLabel(label) {
        if (!this.docs[this.doc_index].completed_by_user) {
          const a = this.isIn(label);
          if (a) {
            this.removeLabel(a);
          } else {
            if (!this.project.multilabel && this.docs[this.doc_index].annotations.length !== 0){
                this.removeLabel(this.docs[this.doc_index].annotations[0]);
            }
            const docId = this.docs[this.doc_index].id;
            const payload = {
              label: label.id,
            };
            await HTTP.post(`docs/${docId}/annotations/`, payload).then((response) => {
              this.docs[this.doc_index].annotations.push(response.data);
            });
          }
          this.fetchProgress();
         }
    },

    labelStyle(label) {
    if (this.use_color){
        if (this.isActiveLabel(label)) {
            return {backgroundColor: label.background_color, color: label.text_color, border: '2px solid ' + label.background_color}
          } else {
            return {border: '2px solid ' + label.background_color, backgroundColor: this.brighter(label.background_color)}
          }
        }
        return {}
    },
  },

});