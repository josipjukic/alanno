import Vue from "vue";
import HTTP from "../http";


const vm = new Vue({
    el: "#dataset",
    delimiters: ["[[", "]]"],

    data: {
        docs: [],
        previous_docs: [],
        next_docs: [],

        isActive: false,
        isDelete: false,
        isDeleteAll: false,

        doc_id: -1,
        project_id: -1,
        expanded_doc_text: '',
        doc_selectors: [],
        doc_completed: [],
        doc_round_number: -1,
        doc_annotations: [],

        searchQuery: '',
        filter_options: ['All', 'In Progress', "Finished"],
        filter: 'All',
        start_index: 0,
        document_count: 0,
        current_page: 0,
        num_pages: 0,
        nextUrl: null,
        prevUrl: null,

    },

    methods: {
        ask(doc_id) {
            this.doc_id = doc_id;
            this.isDelete = true;
        },

        askForAll() {
            this.isDeleteAll = true;
        },

        async deleteDocument() {
            await HTTP.post('delete-documents', { 'doc_id': this.doc_id })
            this.loadPage();
            this.isDelete = false;
        },

        deleteAllDocuments() {
            HTTP.delete('delete-documents').then((r) => {
                this.docs = [];
                this.previous_docs = [];
                this.next_docs = [];
                this.current_page = 0;
                this.num_pages = 0;
                this.isDeleteAll = false;
            })
        },

        expandCell(doc_id) {
            this.docs.forEach((doc) => {
                if (doc.id === doc_id) {
                    this.expanded_doc_text = doc.raw_html;
                    this.doc_selectors = doc.selectors;
                    this.doc_completed = doc.completed;
                    this.doc_round_number = doc.round_number;
                    this.doc_annotations = doc.all_annotations;
                }
            })
            this.doc_id = doc_id;
            this.isActive = true;
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

        // Pagination
        async getNextPage() {
          if (this.nextUrl) {
            await this.fetchDocuments(this.nextUrl);
          }
        },

        async getPrevPage() {
          if (this.prevUrl) {
            await this.fetchDocuments(this.prevUrl);
          }
        },

        async loadPage() {
          await this.fetchDocuments(this.searchURL);
        },

        onPickedChanged(picked) {
          this.filter = picked;
          this.loadPage();
        },

    },


    computed: {
        truncatedDocs: function () {
            const truncated = [];
            var counter = 1;
            this.docs.forEach((doc) => {
                var newDoc = {}
                newDoc.text = doc.text.slice(0, 200);
                newDoc.id = doc.id;
                newDoc.order = counter + this.start_index;
                counter++;
                truncated.push(newDoc)
            })
            return truncated;
        },

        searchURL() {
            return `get-documents/?q=${this.searchQuery}&filter=${this.filter.toLowerCase()}`;
        },

        getDocText: function () {
            return this.expanded_doc_text.replaceAll("\n", "<br/>");
        },

        notEmpty: function () {
            return this.docs.length > 0;
        },

        hasPrevious: function () {
            return this.current_page > 0;
        },

        hasNext: function () {
            return this.current_page + 1 < this.num_pages;
        },

        pageNbr: function () {
            return this.current_page + 1;
        },

        pageMaxNbr: function () {
            return this.num_pages;
        }
    },


    // This method is called when the page is (re)loaded.
    created() {
        this.loadPage()
    }

})

