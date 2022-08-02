import Vue from "vue";
import annotationMixin from "./gl_anno";
import HTTP from "../http";


Vue.use(require("vue-shortkey"), {
  prevent: ["input", "textarea"],
});

const INTERPUNCTION = ['.', ',', '?', '!', ':', ';'];
const NBSP = "\u00A0";

const INACTIVE_STYLE = {
  display: 'inline-block',
  margin: '10px 0 10px 0',
  padding: '0',
  backgroundColor: 'white',
  border: 'none',
  borderRadius: '0',
  fontSize: '1em'
}


const vm = new Vue({
  el: "#main",
  delimiters: ["[[", "]]"],
  mixins: [annotationMixin],
  data: {
    activeLabel: null,
    tokens: [],
    startChar: 0,
    clickedAnnotation: null
  },
  methods: {
    // Token management
    tokenize(text) {
      let tokens = [];
      let temp = [];
      for (let i = 0; i < text.length; i++) {
        let c = text[i];
        if (!(INTERPUNCTION.includes(c) || c === ' ')) {
          temp.push(c);
          continue;
        }
        c = (c === ' ' ? NBSP : c);
        if (temp.length === 0) {
          tokens.push({word: c, start_offset: i, end_offset: i+1, index: tokens.length});
          continue;
        }

        let word = temp.join('');
        tokens.push({word: word, start_offset: i - word.length, end_offset: i, index: tokens.length});
        tokens.push({word: c, start_offset: i, end_offset: i+1, index: tokens.length});
        temp = [];
      }
      if (temp.length > 0) {
        let word = temp.join('');
        tokens.push({word: word, start_offset: text.length - word.length, end_offset: text.length, index: tokens.length});
      }
      this.tokens = tokens;
    },

    getStyleForToken(token) {
      let annos = this.paginated_docs[this.doc_index].annotations[this.username];
      let parentAnnos = annos.filter(anno => token.start_offset >= anno.start_offset && token.end_offset <= anno.end_offset);
      if (parentAnnos.length === 0) {
        return INACTIVE_STYLE;
      }

      let anno = parentAnnos[0];
      let label = this.labels.filter(lbl => lbl.id === anno.label)[0];
      let border = '2px solid ' + label.background_color;
      let radius = '5px';
      let spacing = '2px';

      let style = {
        display: 'inline-block',
        borderTop: border,
        borderBottom: border,
        backgroundColor: this.brighter(label.background_color),
        paddingTop: spacing,
        paddingBottom: spacing,
        fontSize: '1.1em'
      };
      if (anno.start_offset === token.start_offset) {
        style.marginLeft = spacing;
        style.paddingLeft = spacing;
        style.borderLeft = border;
        style.borderBottomLeftRadius = radius;
        style.borderTopLeftRadius = radius;
      }
      if (anno.end_offset === token.end_offset) {
        style.marginRight = spacing;
        style.paddingRight = spacing;
        style.borderRight = border;
        style.borderBottomRightRadius = radius;
        style.borderTopRightRadius = radius;
      }
      return style;
    },

    // Active label management
    setActiveLabel(label) {
      this.activeLabel = label;
    },

    labelStyle(label) {
      if (!this.activeLabel) {
        return {};
      }
      if (label.text === this.activeLabel.text) {
        return {backgroundColor: label.background_color, color: label.text_color, border: '2px solid ' + label.background_color}
      } else {
        return {border: '2px solid ' + label.background_color, backgroundColor: this.brighter(label.background_color)}
      }
    },

    // Selection handling
    onTokenMouseDown(token) {
      if (token.word === "\u00A0") {
        token = this.tokens[token.index + 1];
      }
      this.clickedAnnotation = this.getClickedAnnotation(token);
      this.startChar = token.start_offset;
    },

    onTokenMouseUp(token) {
      this.unselectVanillaSelection();
      if (token.start_offset < this.startChar) {
        return;
      }
      if (token.word === "\u00A0") {
        token = this.tokens[token.index - 1];
      }

      let doc = this.paginated_docs[this.doc_index];
      if (this.clickedAnnotation !== null) {
        // If user clicked DOWN on existing annotation, reject selection
        if (token.start_offset >= this.clickedAnnotation.start_offset && token.end_offset <= this.clickedAnnotation.end_offset) {
          // If they clicked UP on the SAME annotation, delete it
          HTTP.delete(`docs/${doc.id}/annotations-gl/${this.clickedAnnotation.id}`)
            .then(response => {
              doc.is_gl_annotated = response.data.gl_annotated;
            });
          for (let i = 0; i < doc.annotations[this.username].length; i++) {
            if (doc.annotations[this.username][i].id === this.clickedAnnotation.id){
                doc.annotations[this.username].splice(i, 1);
                break;
            }
      }
        }
        return;
      }
      if (this.isRangeAnnotated(this.startChar, token.end_offset)) {
        // If selection overlaps with any other annotation, reject selection
        return;
      }

      // All checks passed: creating the annotation
      HTTP.post(`docs/${doc.id}/annotations-gl/`, {label: this.activeLabel.id, start_offset: this.startChar, end_offset: token.end_offset})
        .then(response => {
          this.paginated_docs[this.doc_index].annotations[this.username].push(response.data);
          this.paginated_docs[this.doc_index].is_gl_annotated = true;
        });
    },

    // Utility
    getClickedAnnotation(token) {
      let result = null;
      let parents = this.paginated_docs[this.doc_index].annotations[this.username].filter(anno => token.start_offset >= anno.start_offset && token.end_offset <= anno.end_offset);
      if (parents.length > 0) {
        result = parents[0];
      }
      return result;
    },

    isRangeAnnotated(start_offset, end_offset) {
      return this.paginated_docs[this.doc_index].annotations[this.username].filter(anno => start_offset <= anno.end_offset && anno.start_offset <= end_offset).length > 0;
    },

    unselectVanillaSelection() {
      if (window.getSelection) {
        window.getSelection().removeAllRanges();
      } else if (document.selection) {
        document.selection.empty();
      }
    },

  },

  watch: {
    labels: function() {
      this.activeLabel = this.labels[0];
    },

    current_doc_id: function() {
      this.tokenize(this.paginated_docs[this.doc_index].text);
    }
  },

});
