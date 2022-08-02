import Vue from "vue";
import annotationMixin from "./annotations";
import HTTP from "../http";


Vue.use(require("vue-shortkey"), {
  prevent: ["input", "textarea"],
});

const INTERPUNCTION = ['.', ',', '?', '!', ':', ';'];
const NBSP = "\u00A0";

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
          tokens.push({word: c, start: i, end: i+1, index: tokens.length});
          continue;
        }

        let word = temp.join('');
        tokens.push({word: word, start: i - word.length, end: i, index: tokens.length});
        tokens.push({word: c, start: i, end: i+1, index: tokens.length});
        temp = [];
      }
      if (temp.length > 0) {
        let word = temp.join('');
        tokens.push({word: word, start: text.length - word.length, end: text.length, index: tokens.length});
      }
      this.tokens = tokens;
    },

    getStyleForToken(token) {
      let annos = this.docs[this.doc_index].annotations;
      let parentAnnos = annos.filter(anno => token.start >= anno.start_offset && token.end <= anno.end_offset);
      if (parentAnnos.length === 0) {
        return {
          display: 'inline-block',
          margin: '10px 0 10px 0',
          padding: '0',
          backgroundColor: 'white',
          border: 'none',
          borderRadius: '0',
          fontSize: '1em'
        }
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
      if (anno.start_offset === token.start) {
        style.marginLeft = spacing;
        style.paddingLeft = spacing;
        style.borderLeft = border;
        style.borderBottomLeftRadius = radius;
        style.borderTopLeftRadius = radius;
      }
      if (anno.end_offset === token.end) {
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
      this.startChar = token.start;
    },

    onTokenMouseUp(token) {
      this.unselectVanillaSelection();
      if (token.start < this.startChar) {
        return;
      }
      if (token.word === "\u00A0") {
        token = this.tokens[token.index - 1];
      }

      let doc = this.docs[this.doc_index];
      if (this.clickedAnnotation !== null) {
        // If user clicked DOWN on existing annotation, reject selection
        if (token.start >= this.clickedAnnotation.start_offset && token.end <= this.clickedAnnotation.end_offset) {
          // If they clicked UP on the SAME annotation, delete it
          HTTP.delete(`docs/${doc.id}/annotations/${this.clickedAnnotation.id}`)
            .then(response => {
              // Locally remove the annotation from the document
              doc.annotations = doc.annotations.filter(anno => anno.id !== this.clickedAnnotation.id);
              this.clickedAnnotation = null;
            });
        }
        return;
      }
      if (this.isRangeAnnotated(this.startChar, token.end)) {
        // If selection overlaps with any other annotation, reject selection
        return;
      }

      // All checks passed: creating the annotation
      HTTP.post(`docs/${doc.id}/annotations/`, {label: this.activeLabel.id, start_offset: this.startChar, end_offset: token.end})
        .then(response => {
          doc.annotations.push(response.data);
        });
    },

    // Utility
    getClickedAnnotation(token) {
      let result = null;
      let parents = this.docs[this.doc_index].annotations.filter(anno => token.start >= anno.start_offset && token.end <= anno.end_offset);
      if (parents.length > 0) {
        result = parents[0];
      }
      return result;
    },

    isRangeAnnotated(start, end) {
      return this.docs[this.doc_index].annotations.filter(anno => start <= anno.end_offset && anno.start_offset <= end).length > 0;
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
      this.tokenize(this.docs[this.doc_index].text);
    }
  },

});
