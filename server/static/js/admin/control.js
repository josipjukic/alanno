import Vue from 'vue';
import HTTP from '../http';
import Multiselect from 'vue-multiselect';
import VueSlider from 'vue-slider-component';
import VueEllipseProgress from 'vue-ellipse-progress';

Vue.component('multiselect', Multiselect);
Vue.component('slider', VueSlider);
Vue.use(VueEllipseProgress);


const vm = new Vue({
  el: '#control',
  delimiters: ['[[', ']]'],
  data: {
    input: '',
    project: Object,
    annotators: [],
    selectedAnnotators: [],
    batch_size: 10,
    anno_per_dp: 1,
    model_anno_threshold: 1,
    test_proportion: 0.2,
    random_train_proportion: 0,
    use_warm_start: true,
    max_anno: 1,
    method: "Round Robin",
    is_weighted: false,
    info: {},
    options: [],
    bsOptions: {},
    apdOptions: {},
    infoKey: 0,
    messages: [],
    messageType: null,
    distributed: 0,
    total: 0,
    hasRound: false,
    roundNumber: -1,
    roundDate: '',
    roundDocuments: -1,
    weights: null,
    weightsOptions: {}
  },

  computed: {
    compiledMarkdown() {
      return marked(this.input, {
        sanitize: true,
      });
    },

    maOptions() {
      var max_val = 1;
      if (this.annotators.length > 0) {
        max_val = this.annotators.length;
      }
      const options = {
        min: 1,
        max: max_val,
        interval: 1,
        duration: 0.5,
        tooltip: 'always',
        tooltipPlacement: 'bottom',
        useKeyboard: true
      }
      return options;
    },

    tpOptions() {
      const options = {
        min: 0,
        max: 1,
        interval: 0.01,
        duration: 0.5,
        tooltip: 'always',
        tooltipPlacement: 'bottom',
        useKeyboard: true
      }
      return options;
    },

  },

  async created() {
    await HTTP.get().then((response) => {
      this.project = response.data;
      this.annotators = this.project.annotators.sort((a, b) => a.username.localeCompare(b.username));
      this.selectedAnnotators = this.annotators;
      var dict = {};
      for (var i = 0; i < this.selectedAnnotators.length; i++) {
          dict[this.selectedAnnotators[i].username] = 1;
      }
      this.weights = dict;
      this.options =
        [
          {
            group: 'All',
            annotators: this.annotators
          },
        ];

      this.bsOptions = {
        min: 1,
        max: 100,
        interval: 1,
        duration: 0.5,
        tooltip: 'always',
        tooltipPlacement: 'bottom',
        useKeyboard: true
      }

      this.weightsOptions = {
        min: 1,
        max: 10,
        interval: 1,
        duration: 0.5,
        tooltip: 'always',
        tooltipPlacement: 'right',
        useKeyboard: true
      }

      this.apdOptions = {
        min: 1,
        max: Math.max(this.project.annotators.length, 1),
        interval: 1,
        duration: 0.5,
        tooltip: 'always',
        tooltipPlacement: 'bottom',
        useKeyboard: true,
        disabled: true
      }
    });

    await HTTP.post('batch-info', this.project.annotators)
      .then(response => {
        this.info = response.data.info;
        console.log("Response: ", response.data);
        console.log("INFO: ", this.info);
        if (response.data.has_round) {
          this.hasRound = response.data.has_round;
          this.roundNumber = response.data.round_number;
          this.roundDate = new Date(response.data.round_date + "Z").toLocaleString(navigator.language);
          this.roundDocuments = response.data.round_documents;
        }
      });

    await HTTP.get('distribution-stats')
      .then(response => {
        this.distributed = response.data.distributed;
        this.total = response.data.total;
      });

  },

  watch: {
    selectedAnnotators: function (newSelection, oldSelection) {
      if (oldSelection) {
        if (this.anno_per_dp > this.selectedAnnotators.length){
            this.anno_per_dp = this.selectedAnnotators.length;
        }
        this.update_apdOptions(newSelection);
        if (oldSelection.length < newSelection.length){
          for (var i = 0; i < this.selectedAnnotators.length; i++) {
            if (!this.weights.hasOwnProperty(this.selectedAnnotators[i].username)){
               this.weights[this.selectedAnnotators[i].username] = 1;
            }
          }
        }else{
          for (var i = 0; i < oldSelection.length; i++) {
            if (!newSelection.includes(oldSelection[i])) {
                delete this.weights[oldSelection[i].username]
            }
          }
        }
      }
    },

    method: function (val) {
      if (this.method === "Weighted Distribution") {
        this.is_weighted = true;
      }else{
        this.is_weighted = false;
      }
    },

  },

  methods: {
    update_apdOptions: _.debounce(function (annotators) {
      if (annotators.length === 0) {
        this.apdOptions.disabled = true;
      } else {
        this.apdOptions.max = annotators.length;
        this.apdOptions.disabled = false;
      }
    }, 300),

    // getInfo(index) {
    //   if (this.info[index]) {
    //     const info = this.info[index];
    //     return `${info.total - info.active}/${info.total}`;
    //   }
    //   return null;
    // },

    getDistributionProgress() {
      if (this.total > 0) {
        const progress = (this.distributed * 100) / this.total;
        return progress;
      }
      return 0;
    },

    getProgress(index) {
      const info = this.info[index];
      if (info && info.total > 0) {
        const done = info.total - info.active;
        const progress = (done * 100) / info.total;
        return progress;
      }
      return 0;
    },

    getCompleted: function (index) {
      const info = this.info[index];
      if (info) {
        const done = info.total - info.active;
        return done;
      }
      return 0;
    },

    getTotal: function (index) {
      const info = this.info[index];
      if (info) {
        return info.total;
      }
      else {
        return 0;
      }
    },

    resetMessages() {
      this.messages = [];
    },

    async generateBatch() {
      const request = {
        "annotators": this.selectedAnnotators.map((a) => a.id),
        "batch_size": this.batch_size,
        "anno_per_dp": this.anno_per_dp,
        "model_anno_threshold": this.model_anno_threshold,
        "test_proportion": this.test_proportion,
        "random_train_proportion": this.random_train_proportion,
        "use_warm_start": this.use_warm_start,
        "method": this.method,
        "weights": this.weights
      }
      await HTTP.post('generate-batch', request)
        .then((response) => {
          this.selectedAnnotators = [];
          this.messages = response.data.messages;
          this.messageType = response.data.message_type;
          HTTP.post('batch-info', this.project.annotators)
            .then(response => {
              this.info = response.data;
            });
        });
      if(this.messageType === "success"){
          window.scrollTo(0,0);
          history.scrollRestoration = 'manual';
          location.reload();
      }
    },


  },

});
