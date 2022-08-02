import Vue from 'vue';
import Vue2OrgTree from 'vue2-org-tree';
import HTTP from '../http';


Vue.component('vue2-org-tree', Vue2OrgTree);

var color_index = 0;
var colors = ["#004949","#009292","#ff6db6","#ffb6db","#490092","#006ddb","#b66dff","#6db6ff","#b6dbff","#920000","#924900","#db6d00","#24ff24"];

function getRandomColor() {
  color_index = (color_index + 1) % colors.length;
  return colors[color_index];
}


const reg = new Vue({
  el: '#reg',
  delimiters: ['[[', ']]'],
  data: {
    labels: [],
    labelText: '',
    selectedShortkey: '',
    backgroundColor: '',
    textColor: '#ffffff',
  },

  methods: {
    addLabel() {
      const payload = {
        text: this.labelText,
        shortcut: this.selectedShortkey,
        background_color: this.backgroundColor,
        alt_color: '#ffffff',
        text_color: this.textColor,
      };
      HTTP.post('labels/', payload)
        .then((response) => {
          this.reset();
          this.labels.push(response.data);
        })
        .catch(err => alert('Invalid operation.'));
    },

    removeLabel(label) {
      const labelId = label.id;
      HTTP.delete(`labels/${labelId}`).then((response) => {
        const index = this.labels.indexOf(label);
        this.labels.splice(index, 1);
      });
    },

    reset() {
      this.labelText = '';
      this.selectedShortkey = '';
      this.backgroundColor = getRandomColor();
      this.textColor = '#ffffff';
    },

    onEnter() {
      this.addLabel();
      document.getElementById("label_name").focus();
    }
  },
  created() {
    HTTP.get('labels').then((response) => {
        this.labels = response.data;
        var color_count = {};
        for (var i=0; i < colors.length; i++){
            color_count[colors[i]] = 0;
        }
        for (var i=0; i < this.labels.length; i++){
            if (this.labels[i].background_color in color_count){
                color_count[this.labels[i].background_color]++;
            }
        }
        for (var i=1; i < colors.length; i++){
            if (color_count[colors[i]] < color_count[colors[i - 1]]){
                color_index = i;
                break;
            }
        }
        this.backgroundColor = colors[color_index];
    });
    //
    // let elements = ["label_name", "label_shortcut", "label_bg", "label_fg"]
    //     .map(id => document.getElementById(id));
    //
    // for (const elem of elements) {
    //   elem.addEventListener("keyup", function(event) { me.onKeyPress(event); });
    // }
  },
});


/*
const hier = new Vue({
  el: '#hier',
  delimiters: ['[[', ']]'],
  data: {
    labels: [],
    labelText: '',
    selectedShortkey: '',
    parent: '',
    isLeaf: false,
    backgroundColor: getRandomColor(),
    textColor: '#ffffff',
    treeData: {},
    horizontal: true,
    collapsable: true,
    expandAll: true,
    labelClassName: "node-block"
  },

  methods: {
    addLabel() {
      const payload = {
        text: this.labelText,
        shortcut: this.selectedShortkey,
        background_color: this.backgroundColor,
        alt_color: '#ffffff',
        text_color: this.textColor,
        is_leaf: this.isLeaf,
        parent: this.parent
      };
      HTTP.post('labels/', payload)
        .then((response) => {
          this.reset();
          this.labels.push(response.data);
          this.backgroundColor = getRandomColor();
        })
        .catch(err => alert('Invalid operation!'));
    },

    async removeLabel(label) {
      const labelId = label.id;
      await HTTP.delete(`labels/${labelId}`);
      await HTTP.get('labels').then((response) => {
        this.labels = response.data;
      });
    },

    reset() {
      this.labelText = '';
      this.selectedShortkey = '';
      this.backgroundColor = getRandomColor();
      this.textColor = '#ffffff';
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
    }
  },
  created() {
    HTTP.get('labels').then((response) => {
      this.labels = response.data;
      var root = { label: 'Labels', expand: true };
      root.children = [];

      for (var i in this.labels) {
        const label = this.labels[i];
        if (!label.parent) {
          var parentNode = { label: label.text, expand: true, children: [], expand: false };
          if (label.is_leaf) {
            parentNode.selected = false;
          }
          for (var j in label.children) {
            const child = label.children[j];
            const childNode = { label: child.text, selected: false };
            parentNode.children.push(childNode);
          }
          root.children.push(parentNode);
        }
      }
      this.treeData = root;
    });

  },

});
*/