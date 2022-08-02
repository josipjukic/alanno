import HTTP from '../http';
import Vue from 'vue';
import VueEllipseProgress from 'vue-ellipse-progress';

Vue.use(VueEllipseProgress);


const annotationMixin = {
  data: {
    // Main data
    project: Object,
    labels: [],
    docs: [],
    doc_index: 0,
    current_doc_id: null,

    // Progress
    batch_total: null,
    batch_remaining: null,

    // Filtering and pagination
    searchQuery: '',
    pick_options: ['All', 'Active', 'Completed'],
    picked: 'All',
    start_index: 0,
    document_count: 0,
    current_page: 0,
    num_pages: 0,
    nextUrl: null,
    prevUrl: null,
    ordering: 'round',

    // Modal data and control
    guidelines: '',
    isActive: false,
    shortcutsActive: false,

    use_color: false,
  },

  methods: {

    // Initialization and fetching
    onPageLoaded() {
      let that = this;
      let ignore = this.pingAnnotationsOpened();

      Promise.all([this.fetchProject(), this.fetchProjectLabels(), this.fetchProjectGuidelines(), this.fetchProgress(), this.whoAmI()])
        .then(ignore => {
          return this.searchDocuments();
        });

      document.addEventListener("visibilitychange", function() {
        if (document.visibilityState === 'visible') {
          that.pingAnnotationsOpened();
          that.pingDocumentOpened(that.docs[that.doc_index].id);
        } else {
          let f1 = false;
          let f2 = false;
          that.pingDocumentClosed(that.docs[that.doc_index].id).then(ignore => {f2 = true;});
          that.pingAnnotationsClosed().then(ignore => {f1 = true;});

          let i = 0;
          while (i < 5e7 || (f1 && f2)) {
            i++;
          }
        }
      });
    },

    async whoAmI() {
      await fetch("/api/whoami")
        .then(response => {
          return response.json();
        }).then(json => {
          this.use_color = json.use_color;
        })
    },


    async fetchProject() {
      HTTP.get().then((response) => {
        this.project = response.data;
      });
    },

    async fetchProjectLabels() {
      await HTTP.get('labels').then((res) => {
        this.labels = res.data;
      });
    },

    async fetchProjectGuidelines() {
      await HTTP.get().then((res) => {
        if (res.data.guidelines) {
          this.guidelines = res.data.guidelines;
        }
        else {
          this.guidelines = '';
        }
      });
    },

    async fetchDocuments(url) {
      await HTTP.get(url).then((response) => {
        this.docs = response.data.results;
        this.nextUrl = response.data.next;
        this.prevUrl = response.data.previous;
        this.document_count = response.data.count;
        this.start_index = response.data.start_index;
        this.current_page = response.data.current_page;
        this.num_pages = response.data.num_pages;
      });
    },

    async fetchProgress() {
      await HTTP.get('progress').then((response) => {
        this.total = response.data.total;
        this.remaining = response.data.remaining;
        this.batch_total = response.data.batch;
        this.batch_remaining = response.data.batch_remaining;
      });
    },

    // Pagination
    async getNextPage() {
      let that = this;
      let oldId = this.docs[this.doc_index].id;

      if (this.nextUrl) {
        this.fetchDocuments(this.nextUrl)
          .then(ignore => {
              that.doc_index = 0;
              let newId = that.docs[that.doc_index].id;

              if (newId !== oldId) {
                this.current_doc_id = newId;
                that.scrollUp();
                that.pingDocumentClosed(oldId).then(ignore2 => {
                  that.pingDocumentOpened(newId);
                });
              }
          });
      }
    },

    async getPrevPage() {
      let that = this;
      let oldId = this.docs[this.doc_index].id;

      if (this.prevUrl) {
        this.fetchDocuments(this.prevUrl)
          .then(ignore => {
              that.doc_index = that.docs.length - 1;
              let newId = that.docs[that.doc_index].id;

              if (newId !== oldId) {
                this.current_doc_id = newId;
                that.scrollUp();
                that.pingDocumentClosed(oldId).then(ignore2 => {
                  that.pingDocumentOpened(newId);
                });
              }
          });
      }
    },

    async nextItem() {
      let that = this;
      let oldId = this.docs[this.doc_index].id;

      if (this.doc_index + 1 >= this.docs.length) {
        await this.getNextPage();
      }
      else {
        this.doc_index += 1;
        this.current_doc_id = this.docs[this.doc_index].id;
        this.pingDocumentClosed(oldId).then(ignore => {
          that.pingDocumentOpened(that.docs[that.doc_index].id);
        });
        this.scrollUp();
      }
    },

    async prevItem() {
      let that = this;
      let oldId = this.docs[this.doc_index].id;

      if (this.doc_index <= 0) {
        await this.getPrevPage();
      }
      else {
        this.doc_index -= 1;
        this.current_doc_id = this.docs[this.doc_index].id;
        this.pingDocumentClosed(oldId).then(ignore => {
          that.pingDocumentOpened(that.docs[that.doc_index].id);
        });
        this.scrollUp();
      }
    },

    async searchDocuments() {
      let that = this;
      let oldId = null;
      if (this.docs.length > 0 && this.docs[this.doc_index]) {
        oldId = this.docs[this.doc_index].id;
      }
      this.fetchDocuments(this.searchURL)
        .then(ignore => {
          that.doc_index = 0;
          let newId = that.docs[0].id;
          if (newId !== oldId) {
            this.current_doc_id = newId;
            that.scrollUp();
            if (oldId) {
              that.pingDocumentClosed(oldId).then(ignore2 => {
                that.pingDocumentOpened(newId);
              });
            } else {
              that.pingDocumentOpened(newId);
            }
          }
        });
    },

    // Event handlers
    onDocumentLock() {
      if (this.docs[this.doc_index].annotations.length !== 0) {
        const doc = this.docs[this.doc_index]
        doc.completed_by_user = !doc.completed_by_user;
        HTTP.post(`docs/${doc.id}/completion`, { completed: doc.completed_by_user })
          .then(() => {
            this.fetchProgress().then((res) => {});
            if (doc.completed_by_user) {
              setTimeout(() => this.nextItem(), 300);
            }
          });
      }
    },

    onPickedChanged(picked) {
      this.picked = picked;
      this.searchDocuments();
    },

    onDocumentClicked(index) {
      if (this.docs[index].id !== this.docs[this.doc_index].id) {
        this.pingDocumentClosed(this.docs[this.doc_index].id);
        this.pingDocumentOpened(this.docs[index].id);
      }
      this.doc_index = index;
      this.current_doc_id = this.docs[index].id;
      this.scrollUp();
    },

    scrollUp() {
      let element = document.getElementById('doc_content');
      if (element) {
        element.scrollTop = 0;
      }
    },

    // Extension points
    isActiveLabel(label) {
      const lab_ids = this.docs[this.doc_index].annotations.map(x => x.label);
      return lab_ids.includes(label.id);
    },

    removeLabel(annotation) {
      const doc = this.docs[this.doc_index];
      HTTP.delete(`docs/${doc.id}/annotations/${annotation.id}`).then((response) => {
        const index = doc.annotations.indexOf(annotation);
        doc.annotations.splice(index, 1);
        this.fetchProgress();
      });
    },

    // User tracking
    pingAnnotationsOpened() {
      return HTTP.post("/log/anno_start", {});
    },

    pingAnnotationsClosed() {
      return HTTP.post("/log/anno_end", {});
    },

    pingDocumentOpened(docId) {
      return HTTP.post("/log/doc_open", {document_id: docId});
    },

    pingDocumentClosed(docId) {
      return HTTP.post("/log/doc_close", {document_id: docId});
    },

    brighter(color) {
      let colorStrings = [
        color.substring(1, 3),
        color.substring(3, 5),
        color.substring(5)
      ];
      let colorValues = colorStrings.map(s => parseInt(s, 16));
      let colorIncrements = colorValues.map(v => Math.round(0.9 * (255 - v)));
      for (let i = 0; i < 3; i++) {
        colorValues[i] = Math.min(255, colorValues[i] + colorIncrements[i]);
      }
      return "#" + colorValues.map(v => Number(v).toString(16)).join('');
    },
  },

  computed: {

    compiledMarkdown() {
      return marked(
        this.guidelines,
        { sanitize: true }
      );
    },

    getDocText() {
      if (this.docs[this.doc_index].raw_html){
        return this.docs[this.doc_index].raw_html.replaceAll("\n", "<br/>");
      }else{
        return null;
      }
    },

    id2label() {
      let id2label = {};
      for (let i = 0; i < this.labels.length; i++) {
        const label = this.labels[i];
        id2label[label.id] = label;
      }
      return id2label;
    },

    searchURL() {
      return `docs/?q=${this.searchQuery}&search=${this.picked.toLowerCase()}&ordering=${this.ordering}`;
    },

    progress() {
      if (this.batch_total > 0) {
        const done = this.batch_total - this.batch_remaining;
        return (done * 100) / this.batch_total;
      }
      return 0;
    },
  },

  watch: {
    use_color: function () {
      HTTP.post('color-options', { 'use_color': this.use_color });
    },

  },

  created() {
    this.onPageLoaded();
  },

};

export default annotationMixin;
