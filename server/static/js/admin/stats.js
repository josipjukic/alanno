import { HorizontalBar, mixins, Doughnut, Line } from 'vue-chartjs';
import Vue from 'vue';
import HTTP from '../http';
import VueApexCharts from 'vue-apexcharts';
import VueEllipseProgress from 'vue-ellipse-progress';

Vue.component('stats-card', {
  delimiters: ['[[', ']]'],
  data: function () {
    return {
      show: true
    };
  },
  props: {
    title: String
  },
  template: `
      <div style="border-radius: 10px; box-shadow: 5px 5px 25px #627a76; margin-bottom: 50px;">
        <h3 style="margin-top: 30px; margin-left: 30px;" v-on:click="show = !show;">[[ title ]]</h3>
        <hr style="margin-right: 0; margin-left: 0; border: 1px solid #33a242;">
        <div v-bind:style="{ display: (show ? 'block' : 'none')  }">
          <slot></slot>
        </div>
      </div>
  `
});

Vue.component('apexchart', VueApexCharts);
Vue.use(VueEllipseProgress);


const { reactiveProp, reactiveData } = mixins;
Vue.component('line-chart', {
  extends: Line,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        scales: {
          yAxes: [{
            id: 'y',
            type: 'linear',
            ticks: {
              suggestedMin: 0.,
              suggestedMax: 1.
            },
            scaleLabel: {
              display: true,
              labelString: 'Performance measure'
            },
          }],
          xAxes: [{
            id: 'x',
            type: 'linear',
            ticks: {
                precision: 0,
            },
            scaleLabel: {
              display: true,
              labelString: '# of labeled data'
            },
          }]
        },
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});


Vue.component('confidence-line', {
  extends: Line,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        scales: {
          xAxes: [{
            id: 'x',
            type: 'linear',
            tickAmount: 'dataPoints',
            ticks: {
                precision: 0,
            },
            scaleLabel: {
              display: true,
              labelString: '# of labeled data'
            },
          }],
          yAxes: [{
            id: 'y',
            type: 'linear',
            position: 'left',
            ticks: {
              suggestedMin: 0.,
              suggestedMax: 1.
            },
            scaleLabel: {
              display: true,
              labelString: 'Performance measure'
            },
          }]
        },
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});

Vue.component('iaa-trend', {
  extends: Line,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        scales: {
          yAxes: [{
            id: 'y',
            type: 'linear',
            ticks: {
              suggestedMin: 0.,
              suggestedMax: 1.
            },
            scaleLabel: {
              display: true,
              labelString: "Fleiss' kappa coefficient"
            },
          }],
          xAxes: [{
            id: 'x',
            type: 'linear',
            ticks: {
                precision: 0,
            },
            scaleLabel: {
              display: true,
              labelString: 'Round number'
            },
          }]
        },
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});

Vue.component('iaa-trend-multilabel', {
  extends: Line,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        scales: {
          yAxes: [{
            id: 'y',
            type: 'linear',
            ticks: {
              suggestedMin: 0.,
              suggestedMax: 1.
            },
            scaleLabel: {
              display: true,
              labelString: "Joint Krippendorff's alpha + MASI distance"
            },
          }],
          xAxes: [{
            id: 'x',
            type: 'linear',
            ticks: {
                precision: 0,
            },
            scaleLabel: {
              display: true,
              labelString: 'Round number'
            },
          }]
        },
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});

Vue.component('horizontal-bar-chart', {
  extends: HorizontalBar,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        dataset: {
          barPercentage: 0.2
        },
        scales: {
          xAxes: [{
            ticks: {
              beginAtZero: true,
              min: 0,
              precision: 0,
            },
          }],
          yAxes: [{
            gridLines: {
              display: false
            }
          }]
        },
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});


Vue.component('doughnut-chart', {
  extends: Doughnut,
  mixins: [reactiveProp],
  props: ['chartData'],
  data() {
    return {
      options: {
        maintainAspectRatio: false,
      },
    };
  },

  mounted() {
    this.renderChart(this.chartData, this.options);
  },
});


function radialBarColors({ value, seriesIndex, w }) {
  let percentage = value/100;
  if (percentage < 0.2) {
    return "#000000";
  } else if (percentage >= 0.2 && percentage < 0.4) {
    return "#0a200d";
  } else if (percentage >= 0.4 && percentage < 0.6) {
    return "#1a5121";
  } else if (percentage >= 0.6 && percentage < 0.8) {
    return "#298235";
  } else if (percentage >= 0.8) {
    return "#33a242";
  }
}

const MAGIC_NON_NULL_NUMBER = 2;

const vm = new Vue({
  el: '#stats',
  delimiters: ['[[', ']]'],
  data: {
    lastUpdated: "never",
    annotators: null,
    progress: null,
    modelTestConfidence: null,
    modelTrain: null,
    forecast: false,
    labelData: null,
    glLabelData: null,
    userData: null,
    activeUserData: null,
    progressData: null,
    iaa: null,
    iaaTrend: null,
    timeStats: null,
    data: Object,
    allRounds: "All",
    selected: "All",
    roundChoices: [],
    errorShown: false,
    heatMapOptions: {
      chart: {
        type: 'heatmap',
      },
      dataLabels: {
        enabled: true,
        formatter: function (val, opts) {
          if (val === MAGIC_NON_NULL_NUMBER) {
            return "";
          } else {
            return val;
          }
        },
      },
      tooltip: {
        enabled: true,
        x: {
          show: false
        },
        y: {
          show: true,
          formatter: function (val, opts) {
            if (val === MAGIC_NON_NULL_NUMBER) {
              return "No common documents";
            } else {
              return val;
            }
          }
        }
      },
      legend: {
        show: false
      },
      plotOptions: {
        heatmap: {
          shadeIntensity: 0,
          reverseNegativeShade: true,
          colorScale: {
            ranges: [
              {
                from: -1.1, to: -0.8, color: '#b5394a', name: 'dreadful'
              },{
                from: -0.8, to: -0.6, color: '#912e3c', name: 'atrocious'
              },{
                from: -0.6, to: -0.4, color: '#6d232e', name: 'awful'
              },{
                from: -0.4, to: -0.2, color: '#491820', name: 'bad'
              },{
                from: -0.2, to: 0, color: '#250d12', name: 'poor'
              },{
                from: 0, to: 0.2, color: "#000000", name: 'very low'
              }, {
                 from: 0.2, to: 0.4, color: "#0a200d", name: 'low'
              }, {
                 from: 0.4, to: 0.6, color: "#1a5121", name: 'medium'
              }, {
                 from: 0.6, to: 0.8, color: "#298235", name: 'high'
              }, {
                 from: 0.8, to: 1.1, color: "#33a242", name: 'very high'
              },
              {
                 from: MAGIC_NON_NULL_NUMBER-0.1, to: MAGIC_NON_NULL_NUMBER+0.1, color: "#ffffff", name: 'missing-value'
              }
            ]
          }
        }
      }
    },
    barChartOptions: {
      chart: {
        type: 'bar',
      },
      fill: {
        colors: ["#33a242"]
      },
      plotOptions: {
        bar: {
          borderRadius: 4,
          horizontal: true,
        }
      },
      dataLabels: {
        enabled: true
      },
      xaxis: {
        categories: []
      },
    },
    radialBarOptions: {
      chart: {
        type: 'radialBar',
      },
      fill: {
        colors: [radialBarColors]
      },
      plotOptions: {
        radialBar: {
          dataLabels: {
            name: {
              fontSize: '22px',
              color: '#000000'
            },
            value: {
              fontSize: '16px'
            },
            total: {
              show: true,
              label: 'Total',
              fontFamily: "montserrat-regular",
              color: "#000000",
              formatter: function (w) {
                return Math.round(w.globals.seriesTotals.reduce((a, b) => {
                  return a + b
                }, 0) / w.globals.series.length) + '%'
              }
            }
          }
        }
      },
      labels: [],
    },
  },

  methods: {

    formatTime(seconds) {
      if (seconds < 60) {
        return seconds.toFixed(2) + "s";
      }

      let minutes = Math.floor(seconds / 60);
      seconds = (seconds % 60).toFixed(2);
      if (minutes < 60) {
        return ""+minutes+"m "+seconds+"s";
      }

      let hours = Math.floor(minutes / 60);
      minutes = (minutes % 60).toFixed(2);
      return "" + hours + "h " + minutes + "m " + seconds + "s";
    },

    format(labels, data){
        if (data == null){
            return null
        }
        var chartData = [];
        for (var i = 0; i < labels.length; i++) {
            chartData.push({
                'x': parseFloat(labels[i]),
                'y': parseFloat(data[i])
            });
        }
        return chartData;
    },

    confidenceChartData(test_data, bootstrap_data, ub, lb, labels, label_test, label_bootstrap,
                        mean_pred, ub_pred, lb_pred, x_pred, label_pred) {
      const res = {
        datasets: [
          {
            label: label_test,
            type: "line",
            backgroundColor: "#33a242",
            borderColor: "#33a242",
            hoverBorderColor: "rgb(0, 0, 0)",
            fill: false,
            tension: 0,
            data: this.format(labels, test_data),
            yAxisID: 'y',
            xAxisID: 'x'
          },
          {
            label: label_bootstrap,
            type: "line",
            backgroundColor: "#0c0c0c",
            borderColor: "#0c0c0c",
            hoverBorderColor: "rgb(0, 0, 0)",
            fill: false,
            tension: 0,
            data: this.format(labels, bootstrap_data),
            yAxisID: 'y',
            xAxisID: 'x'
          },
          {
            label: "Upper bound",
            type: "line",
            backgroundColor: "rgba(61, 173, 76, 0.5)",
            borderColor: "transparent",
            pointRadius: 0,
            fill: 1,
            tension: 0,
            data: this.format(labels, ub),
            yAxisID: 'y',
            xAxisID: 'x'
          },
          {
            label: "Lower bound",
            type: "line",
            backgroundColor: "rgba(61, 173, 76, 0.5)",
            borderColor: "transparent",
            pointRadius: 0,
            fill: 1,
            tension: 0,
            data: this.format(labels, lb),
            yAxisID: 'y',
            xAxisID: 'x'
          }]
      };
      if (mean_pred) {
        var pred_data = [{
            label: label_pred,
            type: "line",
            backgroundColor: "#8c8c8c",
            borderColor: "#8c8c8c",
            hoverBorderColor: "rgb(0, 0, 0)",
            fill: false,
            tension: 0,
            borderDash: [5, 1],
            data: this.format(x_pred, mean_pred),
            yAxisID: 'y',
            xAxisID: 'x'
          },
          {
            label: "Forecast upper bound",
            type: "line",
            backgroundColor: "rgba(140, 140, 140, 0.5)",
            borderColor: "transparent",
            pointRadius: 0,
            fill: 4,
            tension: 0,
            data: this.format(x_pred, ub_pred),
            yAxisID: 'y',
            xAxisID: 'x'
          },
          {
            label: "Forecast lower bound",
            type: "line",
            backgroundColor: "rgba(140, 140, 140, 0.5)",
            borderColor: "transparent",
            pointRadius: 0,
            fill: 4,
            tension: 0,
            data: this.format(x_pred, lb_pred),
            yAxisID: 'y',
            xAxisID: 'x'
          },
        ];
        res.datasets.push(...pred_data)
      }
      return res;
    },

    lineChartData(data, labels, label) {
      const res = {
        datasets: [{
          type: "line",
          tension: 0,
          label: label,
          backgroundColor: "#33a242",
          borderColor: "#33a242",
          data: this.format(labels, data),
          fill: false,
        }],
      };
      return res;
    },

    makeData(data, labels, label, bg_color) {
      const res = {
        labels: labels,
        datasets: [{
          label: label,
          backgroundColor: 'rgba(51, 162, 66, 0.9)',
          pointBorderColor: '#2554FF',
          data: data,
        }],
      };
      return res;
    },

    initStats(){
        var round = this.selected;
        if (this.data.test) {
            this.modelTestConfidence = this.confidenceChartData(
              this.data.test.metric,
              this.data.test.bootstrap,
              this.data.test.ub,
              this.data.test.lb,
              this.data.test.count,
              'F1 (Test)',
              'F1 (Bootstrap)',
              this.data.mean_pred,
              this.data.ub_pred,
              this.data.lb_pred,
              this.data.x_pred,
              'F1 (Forecast)');
          }
          if (this.data.train) {
            this.modelTrain = this.lineChartData(this.data.train.metric, this.data.train.count, 'F1 score - train');
          }
          this.annotators = this.data[round].annotators;
          this.labelData = this.makeData(this.data[round].label.data, this.data[round].label.labels, 'Label count', '#00d1b2');
          if (this.data[round].label.gl_data){
            this.glLabelData = this.makeData(this.data[round].label.gl_data, this.data[round].label.labels, 'Label count', '#00d1b2');
          }else{
            this.glLabelData = null;
          }
          this.iaa = this.data[round].iaa;
          this.timeStats = this.data[round].times;
          this.barChartOptions.xaxis.categories = this.annotators.username;
          this.radialBarOptions.labels = this.annotators.username;
          this.progress = this.data[round].progress;
          if (this.roundChoices.length > 1){
              var iaa_trend_labels = [];
              var iaa_trend_values = [];
              for (var i in this.roundChoices){
                iaa_trend_labels.push(parseInt(i) + 1);
                iaa_trend_values.push(this.data[i].iaa.joint);
              }
              this.iaaTrend = this.lineChartData(iaa_trend_values, iaa_trend_labels, 'Joint Inter-Annotator Agreement');
          }
    },

    fetchData(forceUpdate) {
      HTTP.get(`stats?forceUpdate=${forceUpdate ? 1 : 0}`).then((response) => this.processResponse(response));
    },

    forceUpdate() {
      this.fetchData(true);
    },

    processResponse(response) {
      if (!response.data.finished) {
        this.lastUpdated = "never";
        if (!response.data.task_running) {
          location.reload();
        }
      } else {
        this.data = response.data.stats;
        this.lastUpdated = new Date(response.data.lastUpdated+"Z").toLocaleString(navigator.language);
        this.roundChoices = response.data.stats.rounds;
        this.sanitizeIAANullValues();
        this.initStats();
      }
    },

    sanitizeIAANullValues() {
      const rounds = [...this.roundChoices, "All"];

      for (let i = 0; i < rounds.length; i++) {
        let round = rounds[i];
        let table = this.data[round].iaa;
        if (!table) continue;

        for (let j = 0; j < table.pairwise.length; j++) {
          let row = table.pairwise[j].data;

          for (let k = 0; k < row.length; k++) {
            if (row[k].y === null || row[k].y === undefined) {
              this.data[round].iaa.pairwise[j].data[k].y = MAGIC_NON_NULL_NUMBER;
            }
          }
        }
      }
    }
  },

  watch: {
    selected: function (val) {
      this.initStats();
    },
  },

  created() {
    this.fetchData(false);
  },
});
