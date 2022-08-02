import Vue from "vue";
import axios from "axios";
import Multiselect from 'vue-multiselect';

Vue.component('multiselect', Multiselect);

axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";
const baseUrl = window.location.href
  .split("/")
  .slice(0, 3)
  .join("/");

const vm = new Vue({
  el: "#projects_root",
  delimiters: ["[[", "]]"],
  data: {
    items: [], // TODO: replace with [], ensure refresh
    itemsClf: 0,
    itemsSeqLab: 0,
    itemsKex: 0,
    itemsNer: 0,
    isActive: true,
    isDelete: false,
    project: null,
    allProjects: "No Filter",
    selected: "No Filter",
    errorShown: false,
    loading: false,
    projectChoices: [],
    value: '',
    access_code: null,
    messages: [],
    username: null,
  },

  methods: {
    isManager(p) {
        for (var i = 0; i < p.managers.length; i++) {
            if (p.managers[i].username === this.username) {
                return true;
            }
        }
        return false;
    },

    deleteProject() {
      this.isDelete = false;
      this.loading = true;
      axios
        .delete(`${baseUrl}/api/projects/${this.project.id}/`)
        .then((ignore) => {
          this.isDelete = false;
          const index = this.items.indexOf(this.project);
          this.items.splice(index, 1);
          this.loading = false;
        });
    },

    setProject(project) {
      this.project = project;
      this.isDelete = true;
    },

    matchType(projectType) {
      return this.selected === this.projectChoices[projectType];
    },

    getDaysAgo(dateStr) {
      const updatedAt = new Date(dateStr);
      const currentTm = new Date();

      // difference between days(ms)
      const msDiff = currentTm.getTime() - updatedAt.getTime();

      // convert daysDiff(ms) to daysDiff(day)
      return Math.floor(msDiff / (1000 * 60 * 60 * 24));
    },

    getDaysAgoMsg(dateStr) {
      const daysDiff = this.getDaysAgo(dateStr)

      if (daysDiff < 1) {
        return "Updated less than a day ago.";
      } else if (daysDiff === 1) {
        return "Updated 1 day ago.";
      } else {
        return "Updated " + daysDiff.toString() + " days ago.";
      }
    },

    setIsActive() {
      if (!this.errorShown) {
        this.isActive = true;
        this.errorShown = true;
      }
    },

    getProjectType(type) {
      if (type === 'Classification') {
        return 'CLF';
      } else if (type === 'Sequence Labeling') {
        return 'SEQ_LAB';
      } else if (type === 'Keyphrase Extraction') {
        return 'KEX';
      } else if (type === 'Named Entity Recognition') {
        return 'NER';
      }
    },

    resetMessages() {
      this.messages = [];
    },

    async joinProject() {

      const request = {
        "access_code": this.access_code.trim()
      }
      await axios.post(`${baseUrl}/api/projects/join-project`, request)
        .then((response) => {
          this.messages = response.data.messages;
          this.messageType = response.data.message_type;
          this.getProjects();
        });
    },

    async whoAmI() {
      await fetch("/api/whoami")
        .then(response => {
          return response.json();
        }).then(json => {
            console.log(json);
          this.username = json.username;
        })
    },

    async getProjects() {
      await axios.get(`${baseUrl}/api/projects`).then((response) => {
        this.items = response.data;
        this.itemsClf = 0;
        this.itemsSeqLab = 0;
        this.itemsKex = 0;
        this.itemsNer = 0;
        for (let i = 0; i < this.items.length; i++) {
          if (this.items[i].project_type === 'Classification') {
            this.itemsClf++;
          } else if (this.items[i].project_type === 'Sequence Labeling') {
            this.itemsSeqLab++;
          } else if (this.items[i].project_type === 'Keyphrase Extraction') {
            this.itemsKex++;
          } else if (this.items[i].project_type === 'Named Entity Recognition') {
            this.itemsNer++;
          }
        }
      });
      await axios.get(`${baseUrl}/api/projects/types`).then((response) => {
        this.projectChoices = response.data;
      });
    }

  },

  computed: {
    selectedProjects() {
      const projects = [];
      this.items.forEach((item) => {
        if (
          this.selected === this.allProjects ||
          this.matchType(item.project_type)
        ) {
          projects.push(item);
        }
      });
      return projects;
    },
  },

  created() {
    this.whoAmI();
    this.getProjects();
  },
});
