import Vue from 'vue';
import HTTP from '../http';
import Multiselect from 'vue-multiselect';
import VueSlider from 'vue-slider-component';


Vue.component('multiselect', Multiselect);
Vue.component('slider', VueSlider)

const vm = new Vue({
  el: '#settings',
  delimiters: ['[[', ']]'],
  data: {
    input: '',
    project: Object,
    selectionOptions: [],
    selectedAnnotators: null,
    mainSelectionOptions: [],
    mainAnnotator: null,
    confirmDelete: '',
    deleteMessage: null,
    al_method: '',
    model_name: '',
    token_type: '',
    vectorizer_name: '',
    adjustable_vocab: Boolean,
    vocab_max_size: -1,
    vocab_min_freq: -1,
    vocabMaxOptions: {},
    vocabMinOptions: {},
    init: true,
    timer_size: null,
    timer_freq: null,
    old_freq_slider_value: null,
    old_size_slider_value: null,
    min_ngram: -1,
    max_ngram: -1,
    timer_max_ngram: null,
    timer_min_ngram: null,
    old_max_ngram: null,
    old_min_ngram: null,
    min_ngram_saved: -1,
    max_ngram_saved: -1,
    using_word_vectors: Boolean,
    TORCH_MODELS: ["mlp", "rnn", "lstm", "gru", "bert"],
    RECURSIVE_MODELS: ["rnn", "lstm", "gru"],
    DL_AL: ["core_set", "badge"]
  },

  computed: {
    compiledMarkdown() {
      return marked(this.input, {
        sanitize: true,
      });
    },

    maxNgramOptions() {
        const options_max = {
            min: this.min_ngram,
            step: 1,
            value: this.max_ngram_saved
        }
        return options_max;
    },

    minNgramOptions() {
        const options_min = {
            min: 1,
            max: this.max_ngram,
            step: 1,
            value: this.min_ngram_saved
        }
        return options_min;
    }
  },

  watch: {
    selectedAnnotators: function (newSelection, oldSelection) {
      if (oldSelection) {
        this.updateAnnotators(newSelection);
      }
    },

    mainAnnotator: function (newSelection, oldSelection) {
      let oldMainAnno = (oldSelection && oldSelection.length > 0) ? oldSelection[0] : null;
      let newMainAnno = (newSelection && newSelection.length > 0) ? newSelection[0] : null;
      if (newMainAnno !== oldMainAnno) {
        let submitValue = newMainAnno ? newMainAnno.username : null;
        HTTP.post('main-annotator', {'user': submitValue})
          .then(response => {
            this.project.main_annotator = newMainAnno;
          });
      }
    },

    al_method: function (newValue, oldValue) {
        if(oldValue !== ''){
            HTTP.post('al-method', { 'name': this.al_method });
            if(!this.TORCH_MODELS.includes(this.model_name)){
                this.model_name = "lstm"
            }
        }
    },

    model_name: function (newValue, oldValue) {
        if(oldValue !== ''){
            if(!this.TORCH_MODELS.includes(this.model_name) && this.DL_AL.includes(this.al_method)){
                var readable_al_method = document.getElementById("al_method_select").options[this.al_method].text;
                window.alert(readable_al_method+" AL method can be used only with Deep Learning models.");
                this.model_name = oldValue;
            }
            HTTP.post('model-name', { 'name': this.model_name });
            if(this.RECURSIVE_MODELS.includes(this.model_name)){
                this.vectorizer_name = "emb_matrx"
            }else if(!this.RECURSIVE_MODELS.includes(this.model_name) && this.model_name !== "bert" && this.vectorizer_name === "emb_matrx"){
                this.vectorizer_name = "tf_idf"
            }
        }
    },

    token_type: function (newValue, oldValue) {
        if(oldValue !== ''){
            HTTP.post('token-type', { 'name': this.token_type });
        }
    },

    min_ngram: function(newValue, oldValue) {
        if(oldValue !== -1){
            this.old_min_ngram = this.min_ngram;
            if(this.timer_min_ngram){
                clearTimeout(this.timer_min_ngram);
            }
            this.timer_min_ngram = setTimeout(() => {
                this.timer_min_ngram = null;
                if(this.min_ngram == this.old_min_ngram){
                    this.min_ngram = parseInt(this.min_ngram);
                    if(this.min_ngram <= this.max_ngram){
                        this.min_ngram_saved = this.min_ngram;
                    }else{
                        window.alert("Value of field min ngram can not be bigger then max ngram.");
                        this.min_ngram = this.min_ngram_saved;
                    }
                }
            }, 1000)
        }
    },

    min_ngram_saved: function(newValue, oldValue) {
        if(oldValue !== -1){
            HTTP.post('min-ngram', { 'value': this.min_ngram_saved });
        }
    },

    max_ngram: function(newValue, oldValue) {
        if(oldValue !== -1){
            this.old_max_ngram = this.max_ngram;
            if(this.timer_max_ngram){
                clearTimeout(this.timer_max_ngram);
            }
            this.timer_max_ngram = setTimeout(() => {
                this.timer_max_ngram = null;
                if(this.max_ngram == this.old_max_ngram){
                    this.max_ngram = parseInt(this.max_ngram);
                    if(this.min_ngram <= this.max_ngram){
                        this.max_ngram_saved = this.max_ngram;
                    }else{
                        window.alert("Value of field min ngram can not be bigger then max ngram.");
                        this.max_ngram = this.max_ngram_saved;
                    }
                }
            }, 1000)
        }
    },

    max_ngram_saved: function(newValue, oldValue) {
        if(oldValue !== -1){
            HTTP.post('max-ngram', { 'value': this.max_ngram_saved });
        }
    },

    vectorizer_name: function (newValue, oldValue) {
        if(this.vectorizer_name === "count" || this.vectorizer_name === "tf_idf"){
            this.using_word_vectors = false;
        }else{
            this.using_word_vectors = true;
        }
        if(oldValue !== ''){
            if(this.RECURSIVE_MODELS.includes(this.model_name) && this.vectorizer_name !== "emb_matrx"){
                var readable_model_name = document.getElementById("model_name_select").options[this.model_name].text;
                var readable_vectorizer_name = document.getElementById("vectorizer_name_select").options[this.vectorizer_name].text;
                window.alert(readable_model_name+" model does not work with "+readable_vectorizer_name+" vectorizer.");
                this.vectorizer_name = oldValue;
            }else if(!this.TORCH_MODELS.includes(this.model_name) && this.vectorizer_name === "emb_matrx"){
                var readable_model_name = document.getElementById("model_name_select").options[this.model_name].text;
                var readable_vectorizer_name = document.getElementById("vectorizer_name_select").options[this.vectorizer_name].text;
                window.alert(readable_model_name+" model does not work with "+readable_vectorizer_name+" vectorizer.");
                this.vectorizer_name = oldValue;
            }else if(this.model_name === "mlp" && this.vectorizer_name == "emb_matrx"){
                var readable_model_name = document.getElementById("model_name_select").options[this.model_name].text;
                var readable_vectorizer_name = document.getElementById("vectorizer_name_select").options[this.vectorizer_name].text;
                window.alert(readable_model_name+" model does not work with "+readable_vectorizer_name+" vectorizer.");
                this.vectorizer_name = oldValue;
            }
            HTTP.post('vectorizer-name', { 'name': this.vectorizer_name });
        }
    },

    adjustable_vocab: function (newValue, oldValue) {
        if(this.init){
            this.init = false;
        }else{
            HTTP.post('adjustable-vocab', { 'value': this.adjustable_vocab });
        }
    },

    vocab_max_size: function (newValue, oldValue) {
        if(oldValue !== -1){
            this.old_size_slider_value = this.vocab_max_size;
            if(this.timer_size){
                clearTimeout(this.timer_size);
            }
            this.timer_size = setTimeout(() => {
                this.timer_size = null;
                if(this.vocab_max_size == this.old_size_slider_value){
                    HTTP.post('vocab-max-size', { 'value': this.vocab_max_size });
                }
            }, 1000)
        }
    },

    vocab_min_freq: function (newValue, oldValue) {
        if(oldValue !== -1){
            this.old_freq_slider_value = this.vocab_min_freq;
            if(this.timer_freq){
                clearTimeout(this.timer_freq);
            }
            this.timer_freq = setTimeout(() => {
                this.timer_freq = null;
                if(this.vocab_min_freq == this.old_freq_slider_value){
                    HTTP.post('vocab-min-freq', { 'value': this.vocab_min_freq });
                }
            }, 1000)
        }
    },
  },

  created() {
    HTTP.get().then((response) => {
      this.project = response.data;
      this.input = response.data.guidelines;
      this.selectedAnnotators = this.project.annotators;
      this.mainAnnotator = this.project.main_annotator ? [this.project.main_annotator] : null;
      this.al_method = this.project.al_method;
      this.model_name = this.project.model_name;
      this.token_type = this.project.token_type;
      this.vectorizer_name = this.project.vectorizer_name;
      this.adjustable_vocab = this.project.adjustable_vocab;
      this.vocab_max_size = this.project.vocab_max_size;
      this.vocab_min_freq = this.project.vocab_min_freq;
      this.min_ngram = this.project.min_ngram;
      this.max_ngram = this.project.max_ngram;
      this.min_ngram_saved = this.project.min_ngram;
      this.max_ngram_saved = this.project.max_ngram;

      this.vocabMaxOptions = {
          min: 1000,
          max: 100000,
          interval: 500,
          duration: 0.5,
          tooltip: 'always',
          tooltipPlacement: 'bottom',
          useKeyboard: true,
          value: this.vocab_max_size
      }

      this.vocabMinOptions = {
          min: 1,
          max: 25,
          interval: 1,
          duration: 0.5,
          tooltip: 'always',
          tooltipPlacement: 'bottom',
          useKeyboard: true,
          value: this.vocab_min_freq
      }
    });

      HTTP.get('all-members').then((response) => {
        this.selectionOptions = response.data;
        this.mainSelectionOptions = this.selectionOptions[0].annotators;
      });
  },

  methods: {

    updateGuidelines: _.debounce(function (e) {
      this.input = e.target.value;
      this.project.guidelines = this.input;
      HTTP.post('guidelines', { 'guidelines': this.input });
    }, 300),

    updateAnnotators: _.debounce(function (annotators) {
      this.project.annotators = annotators;
      const annotator_ids = annotators.map((a) => a.id);
      HTTP.post('edit-annotators', annotator_ids);
    }, 300),

    deleteProject() {
      if (this.project.name == this.confirmDelete) {
        HTTP.delete('delete-project').then((r) => {
            window.location.replace("/projects");
        })
      }
      else {
        this.deleteMessage = "Incorrect project name!";
      }
    },

    resetMessage() {
      this.deleteMessage = null;
    }

  },

});
