const VueLoaderPlugin = require('vue-loader/lib/plugin')

module.exports = {
    mode: 'development',
    entry: {
        'document_classification': './static/js/annotation/document_classification.js',
        'hier_classification': './static/js/annotation/hier_classification.js',
        'sequence_labeling': './static/js/annotation/sequence_labeling.js',
        'seq2seq': './static/js/annotation/seq2seq.js',
        'gl_classification': './static/js/gl_anno/gl_classification.js',
        'gl_seq_lab': './static/js/gl_anno/gl_seq_lab.js',
        'stats': './static/js/admin/stats.js',
        'label': './static/js/admin/label.js',
        'dataset': './static/js/admin/dataset.js',
        'control': './static/js/admin/control.js',
        'upload': './static/js/admin/upload.js',
        'settings': './static/js/admin/settings.js',
        'projects': './static/js/projects.js',
        'test': './static/js/test.js',
        'instructions': './static/js/admin/instructions.js'
    },
    output: {
        path: __dirname + '/static/bundle',
        filename: '[name].js'
    },
    watchOptions: {
        ignored: /node_modules/,
        aggregateTimeout: 300,
        poll: 500
      },
    module: {
        rules: [
            {
                test: /\.vue$/,
                loader: 'vue-loader'
            }
        ]
    },
    plugins: [
        new VueLoaderPlugin()
    ],
    resolve: {
        extensions: ['.js', '.vue'],
        alias: {
            vue$: 'vue/dist/vue.esm.js',
        },
    },
}
