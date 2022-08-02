import Vue from 'vue';
import annotationMixin from './gl_anno';
import HTTP from '../http';

Vue.use(require('vue-shortkey'), {
  prevent: ['input', 'textarea'],
});


const vm = new Vue({
  el: '#main',
  delimiters: ['[[', ']]'],
  mixins: [annotationMixin],

  methods: {

    async toggleLabel(label) {
      if (!this.documentAnnotators.includes(this.username)) {
        this.paginated_docs[this.doc_index].annotations[this.username] = [];
      }
      let adminAnnotations = this.paginated_docs[this.doc_index].annotations[this.username];
      let matching = adminAnnotations.filter(anno => anno.label === label.id);
      if (matching.length > 0) {
        await this.removeLabel(matching[0]);
        return;
      }
      if (adminAnnotations.length > 0 && !this.project.multilabel) {
        await this.removeLabel(adminAnnotations[0]);
      }
      await HTTP.post(`docs/${this.current_doc_id}/annotations-gl/`, {label: label.id}).then((response) => {
         this.paginated_docs[this.doc_index].annotations[this.username].push(response.data);
         this.paginated_docs[this.doc_index].is_gl_annotated = true;
      });
    },

    async removeLabel(annotation) {
      const doc = this.paginated_docs[this.doc_index];
      for (let i = 0; i < doc.annotations[this.username].length; i++) {
        if (doc.annotations[this.username][i].id === annotation.id){
            doc.annotations[this.username].splice(i, 1);
            break;
        }
      }
      return HTTP.delete(`docs/${doc.id}/annotations-gl/${annotation.id}`).then((response) => {
        doc.is_gl_annotated = response.data.gl_annotated;
      });
    },

    labelStyle(label) {
        if (this.use_color){
            if (this.isActiveLabel(label)) {
                return {backgroundColor: label.background_color, color: label.text_color, border: '2px solid ' + label.background_color}
            }else {
                return {border: '2px solid ' + label.background_color, backgroundColor: this.brighter(label.background_color)}
            }
        }
        return {}
    },
  },

});