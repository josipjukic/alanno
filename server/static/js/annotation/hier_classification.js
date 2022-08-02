import Vue from 'vue';
import annotationMixin from './annotations';
import Vue2OrgTree from 'vue2-org-tree';
import HTTP from '../http';

Vue.component('vue2-org-tree', Vue2OrgTree);

Vue.use(require('vue-shortkey'), {
  prevent: ['input', 'textarea'],
});


const vm = new Vue({
  el: '#main',
  delimiters: ['[[', ']]'],
  mixins: [annotationMixin],

  data: {
    treeData: {},
    horizontal: true,
    collapsable: true,
    expandAll: true,
    labelClassName: "node-block"
  },

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
      const a = this.isIn(label);
      if (a) {
        this.removeLabel(a);
      } else {
        const docId = this.docs[this.doc_index].id;
        const payload = {
          label: label.id,
        };
        await HTTP.post(`docs/${docId}/annotations/`, payload).then((response) => {
          this.docs[this.doc_index].annotations.push(response.data);
        });
      }

      this.fetchProgress();
    },

    removeLabelHier(label) {
      const annos = this.docs[this.doc_index].annotations;
      var annotation;
      for (var i in annos) {
        const anno = annos[i];
        if (anno.label === label.id) {
          annotation = anno;
          break;
        }
      }

      const doc = this.docs[this.doc_index]
      const docId = this.docs[this.doc_index].id;
      HTTP.delete(`docs/${docId}/annotations/${annotation.id}`).then((response) => {
        const index = doc.annotations.indexOf(annotation);
        doc.annotations.splice(index, 1);
        this.fetchProgress();

      });
    },

    renderContent(h, data) {
      return data.label;
    },
    onExpand: function (e, data) {
      if ('expand' in data) {
        data.expand = !data.expand;

        if (!data.expand && data.children) {
          this.collapse(data.children);
        }
      } else {
        this.$set(data, "expand", true);
      }
    },
    onNodeClick(e, data) {
      if (data.selected != null) {
        data.selected = !data.selected;
        const labInfo = { 'text': data.label, 'id': data.id };
        if (data.selected) {
          this.addLabel(labInfo);
        }
        else {
          this.removeLabelHier(labInfo);
        }

      }
    },
    collapse(list) {
      var _this = this;
      list.forEach(function (child) {
        if (child.expand) {
          child.expand = false;
        }
        child.children && _this.collapse(child.children);
      });
    },
    expandChange() {
      this.toggleExpand(this.data, this.expandAll);
    },
    toggleExpand(data, val) {
      var _this = this;
      if (Array.isArray(data)) {
        data.forEach(function (item) {
          _this.$set(item, "expand", val);
          if (item.children) {
            _this.toggleExpand(item.children, val);
          }
        });
      } else {
        this.$set(data, "expand", val);
        if (data.children) {
          _this.toggleExpand(data.children, val);
        }
      }
    },

    prepareTree() {
      var root = { label: 'Labels', expand: true, children: [] };

      const annos = this.docs[this.doc_index].annotations;
      var anno_lab_ids = [];
      for (var i in annos) {
        const anno = annos[i];
        anno_lab_ids.push(anno.label);
      }

      for (var i in this.labels) {
        const label = this.labels[i];
        if (!label.parent) {
          var parentNode = { id: label.id, label: label.text, expand: false, children: [] };
          if (label.is_leaf) {
            parentNode.selected = anno_lab_ids.includes(parentNode.id);
          }
          for (var j in label.children) {
            const child = label.children[j];
            const childNode = { id: child.id, label: child.text, selected: anno_lab_ids.includes(child.id), children: [] };
            parentNode.children.push(childNode);
          }
          root.children.push(parentNode);
        }
      }
      console.log(root);
      this.treeData = root;
    },

    async submit() {
      console.log('HELLO');
      const state = this.picked;
      this.url = `docs/?q=${this.searchQuery}&search=${state}&ordering=id`;
      await this.search();
      this.doc_index = 0;
      HTTP.get('labels')
        .then((response) => {
          this.labels = response.data;
          this.prepareTree();
        });
    },

    refreshData() {
      this.prepareTree();
    }
  },

});