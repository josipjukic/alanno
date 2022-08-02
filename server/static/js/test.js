import Vue from 'vue';
import Vuetify from 'vuetify';

Vue.use(Vuetify);

const vm = new Vue({
  el: '#test',
  delimiters: ['[[', ']]'],
  data: {
    ex1: { label: 'color', val: 25, color: 'orange darken-3' },
    ex2: { label: 'track-color', val: 75, color: 'green lighten-1' },
    ex3: { label: 'thumb-color', val: 50, color: 'red' },
  }
});