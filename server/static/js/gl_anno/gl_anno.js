import HTTP from '../http';
import Vue from 'vue';
import VueEllipseProgress from 'vue-ellipse-progress';

const annotationMixin = {
  data: {
    // Main data
    project: Object,
    labels: [],
    docs: [],
    doc_index: 0,
    current_doc_id: null,
    username: null,
    paginated_docs: [],

    // Filtering and pagination
    searchQuery: '',
    current_page: 1,
    num_pages: 1,
    start_index: 1,
    page_size: 1,

    // Modal data and control
    guidelines: '',
    isActive: false,
    shortcutsActive: false,

    use_color: false,
  },

  methods: {

    // Initialization and fetching
    onPageLoaded() {
      Promise.all([this.fetchProject(), this.fetchProjectLabels(), this.whoAmI()]);
    },

    async fetchProject() {
      HTTP.get().then((response) => {
        this.project = response.data;
        if (response.data.guidelines) {
          this.guidelines = response.data.guidelines;
        } else {
          this.guidelines = 'No guidelines provided for this project.';
        }
      });
    },

    async fetchProjectLabels() {
      await HTTP.get('labels').then((res) => {
        this.labels = res.data;
      });
    },

    async fetchDocuments(url) {
      await HTTP.get(url).then((response) => {
        this.docs = response.data.current;
        this.current_page = response.data.page;
        this.num_pages = response.data.maxPage;
        this.page_size = response.data.pagination;
        this.paginate();
      });
    },

    async paginate() {
        this.paginated_docs = this.docs.slice(((this.current_page - 1) * this.page_size), (this.current_page * this.page_size));
    },

    async whoAmI() {
      await fetch("/api/whoami")
        .then(response => {
          return response.json();
        }).then(json => {
            console.log(json);
          this.username = json.username;
          this.use_color = json.use_color;
        })
    },

    // Document navigation
    async getNextPage() {
      if (this.paginated_docs.length > 0){
          let that = this;
          let oldId = this.paginated_docs[this.doc_index].id;

          if (this.current_page < this.num_pages) {
              this.current_page += 1;
              this.paginate();
              that.doc_index = 0;
              that.start_index = that.page_size * (that.current_page - 1) + 1;
              if (that.paginated_docs.length > 0) {
                let newId = that.paginated_docs[that.doc_index].id;
                if (newId !== oldId) {
                  this.current_doc_id = newId;
                  that.scrollUp();
                }
              }
          }
      }
    },

    async getPrevPage() {
      if (this.paginated_docs.length > 0){
          let that = this;
          let oldId = this.paginated_docs[this.doc_index].id;

          if (this.current_page > 1) {
              this.current_page -= 1;
              this.paginate();
              that.start_index = that.page_size * (that.current_page - 1) + 1;
              if (that.paginated_docs.length > 0) {
                that.doc_index = that.paginated_docs.length - 1;
                let newId = that.paginated_docs[that.doc_index].id;
                if (newId !== oldId) {
                  this.current_doc_id = newId;
                  that.scrollUp();
                }
              }
          }
      }
    },

    async nextItem() {
      if (this.doc_index + 1 >= this.paginated_docs.length) {
        await this.getNextPage();
      }
      else {
        this.doc_index += 1;
        this.current_doc_id = this.paginated_docs[this.doc_index].id;
        this.scrollUp();
      }
    },

    async prevItem() {
      if (this.doc_index <= 0) {
        await this.getPrevPage();
      }
      else {
        this.doc_index -= 1;
        this.current_doc_id = this.paginated_docs[this.doc_index].id;
        this.scrollUp();
      }
    },

    async searchDocuments() {
      let that = this;
      let oldId = null
      if (this.paginated_docs.length > 0 && this.paginated_docs[this.doc_index]) {
        oldId = this.docs[this.doc_index].id;
      }
      this.fetchDocuments(this.searchURL)
        .then(ignore => {
          that.doc_index = 0;
          that.start_index = 1;
          if (that.paginated_docs.length > 0) {
              let newId = that.paginated_docs[0].id;
              if (newId !== oldId) {
                this.current_doc_id = newId;
                that.scrollUp();
              }
          }
        });
    },

    onDocumentClicked(index) {
      this.doc_index = index;
      this.current_doc_id = this.paginated_docs[index].id;
      this.scrollUp();
    },

    // Extension points
    labelClass(label) {
      if (this.paginated_docs.length > 0) {
        let annos = this.paginated_docs[this.doc_index].annotations;
        if (annos[this.username] && annos[this.username].map(anno => anno.label).includes(label.id)) {
          return 'btn--primary';
        }
      }
      return 'btn-outline-primary';
    },

    scrollUp() {
      let element = document.getElementById('doc_content');
      if (element) {
        element.scrollTop = 0;
      }
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

    isActiveLabel(label) {
      if (this.paginated_docs.length > 0) {
        let annos = this.paginated_docs[this.doc_index].annotations;
        if (annos[this.username] && annos[this.username].map(anno => anno.label).includes(label.id)) {
          return true;
        }
      }
      return false;
    },

  },

  watch: {
    use_color: function () {
      HTTP.post('color-options', { 'use_color': this.use_color });
    },

  },

  computed: {

    isDocumentGlAnnotated() {
      if (this.paginated_docs.length > 0) {
        return this.paginated_docs[this.doc_index].is_gl_annotated;
      } else {
        return false;
      }
    },

    isDocumentSelected() {
      if (this.paginated_docs.length > 0) {
        return this.paginated_docs[this.doc_index].is_selected;
      } else {
        return false;
      }
    },

    isDocumentAnnotated() {
      if (this.paginated_docs.length > 0) {
        return this.documentAnnotators.length > 1;
      } else {
        return false;
      }
    },

    documentAnnotators() {
      if (this.paginated_docs.length > 0) {
        return Object.keys(this.paginated_docs[this.doc_index].annotations);
      } else {
        return [];
      }
    },

    compiledMarkdown() {
      return marked(
        this.guidelines,
        { sanitize: true }
      );
    },

    getDocText() {
      if (this.paginated_docs[this.doc_index].raw_html){
        return this.paginated_docs[this.doc_index].raw_html.replaceAll("\n", "<br/>");
      }else{
        return null;
      }
    },

    searchURL() {
      return `retrieve-documents?query=${this.searchQuery}`;
    },
  },

  created() {
    this.onPageLoaded();
  },

};

export default annotationMixin;
