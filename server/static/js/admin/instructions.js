import Vue from 'vue';


Vue.component('instr-block', {
  delimiters: ['[[', ']]'],
  props: {
    index: Number,
    step: Object
  },
  template: `
    <div style="position: relative; width: 100%;" v-bind:style="{ background: (index % 2 == 0 ? 'white' : 'black') }">
      <!-- GRAPHIC DESIGN -->
      <div style="position: absolute; background: #39b54a; width: 4em; height: 100%; left: 8em"></div>
      <div style="position: absolute; background: #39b54a; width: 10em; height: 10em; top: 5em; left: 5em; border-radius: 50%; display: flex; flex-direction: row; justify-content: center; align-items: center">
        <h1 style="margin: 0; padding: 0; color: white; font-size: 6rem">[[ index + 1 ]]</h1>
      </div>
      
      <!-- CONTENT -->
      <div style="margin-left: 20em; padding-right: 2em; min-width: 400px; max-width: 100%; margin-top: 8em; display: flex; flex-direction: column; justify-content: start; align-items: start">
        <h1 style="margin: 0; padding: 0;" v-bind:style="{ color: (index % 2 == 0 ? 'black' : 'white') }">[[ step.title ]]</h1>
        <h4 style="margin: 10px 0 0; color: #39b54a">[[ step.subtitle ]]</h4>
        <div style="margin-top: 1.5em; margin-bottom: 4em;" v-bind:style="{ color: (index % 2 == 0 ? '#757575;' : '#aaaaaa') }" v-html="step.text"/>
      </div>
    </div>
  `
});

const vm = new Vue({
  el: '#instructions',
  delimiters: ['[[', ']]'],
  data: {
    instructionSteps: [
      {
        title: "Create a project",
        subtitle: "You need to get some work",
        text:
            "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
            "<div style='width: 350px; flex-grow: 1'>" +
            "As an administrator, you get to create new projects and be their manager. Creating a project is the necessary first step before " +
            "any work is done. To create a project, fill in <a href='/projects#create'>the project creation form </a> on the projects page (you can " +
            "always reach this page via the links in the sidebar). " +
            "Once you name and describe your project, you'll need to declare its type and language. The type refers to the NLP problem for which " +
            "we are annotating training data, while the language refers to the language of the documents in the dataset, and not the UI, unfortunately." +
            "</div>" +
            "<div style='width: 350px; flex-grow: 1'>" +
            "Scratch below the surface, and into the whole point of this application, by toggling the <b>Active Learning</b> switch on the form, " +
            "which will display a whole new set of settings for your project. Every Active Learning method relies on an AI model to query the " +
            "dataset in some way. In other words, if you want to use Active Learning, you will need to define the supporting NLP model. These " +
            "settings are more technical in nature and require you to know a bit about NLP and Active Learning. Don't worry, though: you can change " +
            "them anytime on the project Settings page." +
            "</div>" +
            "<div style='width: 350px; flex-grow: 1'>" +
            "As a regular user, and by this we mean annotator, your only option is to be invited to join a project by a project's moderator. " +
            "They will do this by sending you an access code, which you can then safely type into <a href='/projects#join'>this form</a> " +
            "on the bottom of the page. Once you click <b>Join</b>, you are officially a member of the project, and soon enough you will " +
            "be able to start <a href='#5'> annotating documents</a>." +
            "</div>" +
            "</div>"
      },
      {
        title: "Add members",
        subtitle: "We don't want it to be lonely here",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 350px; flex-grow: 1'>" +
          "Going back to the <a href='/projects'>projects page</a>, we now see that apart from the forms for creating and joining a project, it " +
          "also shows all the projects you're a member of. On the surface, clicking on a project just takes you to its Settings page. But actually, " +
          "you are now in the project's \"domain\" - everything you do is related to this particular project, until you eventually go back to the " +
          "projects page, or the homepage. You can verify this by opening the sidebar - it will show several new, project-related options (starting " +
          "with <b>Settings</b>)." +
          "</div>" +
          "<div style='width: 350px; flex-grow: 1'>" +
          "Let's stay on the Settings page for a while, since we're here. The first, and most important part of this page is the project access code. " +
          "You need this to invite other people to join your project as annotators. As described in the <a href='#0'>Create a project</a> section, " +
          "send this code to anyone who wishes to join the project. They will have to create an account if they don't have it already, and then " +
          "paste the access code into the <a href='/projects/#join'>project joining form</a>. After you import a dataset, you will be able to distribute " +
          "documents for annotation to all the members you added." +
          "</div>" +
          "<div style='width: 350px; flex-grow: 1'>" +
          "Apart from the access code, the Settings page provides all the project information, as well as the ability to help your annotators by adding project " +
          "guidelines. If you enabled Active Learning for your project, the project information is followed by, as promised, a section where you can change " +
          "all the related settings. Finally, there is an option to delete the project, in case something goes wrong with it. But please come back!" +
          "</div>" +
          "</div>"
      },
      {
        title: "Define labels",
        subtitle: "But only if you feel like it",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 500px; flex-grow: 1'>" +
          "If a part of your dataset has already been labeled, it might make sense to define the set of labels before importing the dataset. " +
          "That way the labels will be recognized and stored when the dataset is imported. This is especially useful when Active Learning is " +
          "used, because then you can use these documents to enable a <b>\"warm start\"</b> for the supporting model, increasing the efficiency " +
          "of the whole process." +
          "</div>" +
          "<div style='width: 500px; flex-grow: 1'>" +
          "Label creation itself is fairly straightforward - you can choose the label name, its shortcut key (the annotation process can be done " +
          "using just the keyboard, which makes things a bit faster), and its background and text colors. Next to the label creation form you will " +
          "see a list of all the labels you defined for the project so far. These are the options which will be given to annotators when they " +
          "annotate documents." +
          "</div>" +
          "</div>"
      },
      {
        title: "Import a dataset",
        subtitle: "Finally something to work with",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 350px; flex-grow: 1;'>" +
          "In order to annotate documents, we first need to have some documents to annotate. To import a dataset, open the Import page within the " +
          "project's domain. Some basic instructions are written there as well, but it's certainly worth repeating: <br/>" +
          "<ul>" +
          "<li>You can only import CSV files</li>" +
          "<li>The CSV file must contain a <a href=''></a> header row</li>" +
          "<li>Supported headers are: <code style='color: black'>text</code>, <code style='color: black'>document_id</code>, " +
          "<code style='color: black'>label</code> and <code style='color: black'>annotator</code></li>" +
          "<li>Only the <code style='color: black;'>text</code> column is required</li>" +
          "<li>The text may include line breaks <code style='color: black'>\\n</code> and/or HTML tags, which will be used to properly format the text during annotation</li>" +
          "</ul>" +
          "</div>" +

          "<div style='width: 350px; flex-grow: 1;'>" +
          "We know that you may already have a partially annotated dataset, so we made sure you don't lose those annotations. " +
          "You can simply add the <code style='color: black'>annotator</code> and <code style='color: black'>label</code> " +
          "headers, and fill in the values for all pre-annotated documents. There are several constraints and conditions, " +
          "though: <ul>" +
          "<li>If the label isn't present or isn't defined in the project, the annotation is ignored</li>" +
          "<li>If the annotator doesn't correspond to a user's username, the label will be added as an anonymous annotation</li>" +
          "<li>Anonymous annotations are used to enable a warm start for the supporting model for Active Learning</li>" +
          "<li>Currently, importing annotations is only supported for classification projects</li>" +
          "</ul>" +
          "</div>" +

          "<div style='width: 350px; flex-grow: 1;'>" +
          "The <code style='color: black;'>document_id</code> column should contain a numerical identifier of the document, used " +
          "if the same document appears multiple times in the dataset. This can happen in a single-label setup " +
          "(multiple users annotated a document) and in a multi-label setup (a single user gave multiple labels to a document). " +
          "With a document ID, we can recognize this and correctly store all the annotations. If your dataset is not " +
          "pre-annotated or doesn't contain multiple rows for a single document, you don't really need this column." +
          "</div>" +

          "</div>"
      },
      {
        title: "Distribute documents",
        subtitle: "Sharing is caring",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 500px; flex-grow: 1;'>" +
          "You can verify that your data has been successfully uploaded by visiting the project's Data page. Now is the " +
          "time to prepare the stage for the work to come, i.e. the annotating process. The annotators are not independent " +
          "of you, they can't just open any document they want and annotate it - that would defeat the purpose of Active " +
          "Learning. Instead, the system decides which documents will get distributed, and you decide when and to whom. " +
          "In Alanno, documents are distributed in rounds, where each round is independent of others. This way, you can " +
          "adjust the workload for each round, depending on the availability of your annotators." +
          "</div>" +

          "<div style='width: 500px; flex-grow: 1;'>" +
          "Rounds (also called batches) for a project are generated on the project's Control page. When you visit the " +
          "Control page, you will be greeted with a form where you can specify the settings of the round to be generated, " +
          "such as the annotators included in the round, the number of annotators per document, and, of course, the number " +
          "of documents to be annotated in this round. In case Active Learning is enabled, you also get to define additional " +
          "settings, such as the proportion of random data to be annotated (data <b>not selected</b> by an AL strategy), " +
          "the proportion of test data, and the possibility to enable warm-starting the supporting model. " +
          "</div>" +

          "</div>"
      },
      {
        title: "Annotate data",
        subtitle: "We humans sure like to stick labels",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 300px; flex-grow: 1;'>" +
          "Regardless of your role on a project, once the project manager has distributed documents to you, you can start annotating them. " +
          "If you're the manager, you will need to open the Annotation page within the project's domain. Otherwise, annotating is the only " +
          "thing you can do, so it suffices to just enter the project's domain. The annotation process is the core of Alanno, and we tried " +
          "to make it as user-friendly as possible. " +
          "</div>" +

          "<div style='width: 375px; flex-grow: 1;'>" +
          "On the left side of the page you will see your progress in this round, followed by a list " +
          "of documents that were distributed to you. You can filter these documents by their completion status, as well as search them for " +
          "keywords. The bottom of the page contains a lock button sandwiched between navigation buttons. The former is used to submit your " +
          "annotation, while the latter are used to quickly go from one document to the next (or previous). As long as the round is active " +
          "(i.e. until the manager creates a new round), you are free to unlock, change and lock the annotations you provided." +
          "</div>" +

          "<div style='width: 375px; flex-grow: 1;'>" +
          "The center of this page will always display the document itself, with any formatting that was provided as HTML tags during " +
          "<a href='#3'>dataset import</a>. However, the actual manner of annotation depends on the project type and other modifiers. " +
          "This will always be explained on the annotation page itself, but just in case: classification projects require you to select " +
          "one (single-label) or more (multi-label) labels shown below the document, while sequence labeling projects " +
          "require you to select a part of text <b>and</b> a label to apply to the selected text." +
          "</div>" +

          "</div>"
      },
      {
        title: "Track annotation progress",
        subtitle: "Make sure there are no slackers",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 450px; flex-grow: 1;'>" +
          "It is very important to know how your project is doing, and more precisely, how the annotators are managing with the workload. " +
          "To this end, we created a Stats page within the project's domain where you can monitor various kinds of metrics related to your " +
          "project, together with pretty visualisations. The statistics are updated periodically, in case a change has been detected. However, " +
          "we know you may want the data sooner, so we enabled you to force a statistics recalculation. This is especially useful when you think " +
          "the statistics are lagging behind the real situation. It is important to note that the statistics are round-sensitive, meaning the data " +
          "and graphs you see are referring only to the specific round you selected at the top of the page. Alternatively, you can also select " +
          "'All rounds' to show statistics of the project's whole lifecycle. " +
          "</div>" +

          "<div style='width: 550px; flex-grow: 1;'>" +
          "The provided statistics are somewhat grouped into cards with common semantics. The Annotation Progress card displays the current annotator " +
          "progress in the selected round. It shows an overall progress, followed by an individual progress indicator, for each annotator on the " +
          "project. Following this is the Distribution progress card, displaying the number of documents that have already been distributed. This is " +
          "a kind of a 'global progress', since the project is definitely finished once all the documents have been distributed and annotated. Next " +
          "to this is the Label frequency card, showing a histogram of label frequencies in the selected round. Finally, since Alanno is mostly used " +
          "in a multi-annotator setup, it is necessary to provide insight into the Inter Annotator Agreement. We display a visualisation of both " +
          "pairwise and average agreement per user, using Cohen's kappa for single-label setups, and Fleiss' kappa for multi-label ones. If Active " +
          "Learning is enabled, after a few rounds you will also see graphs showing the train and test performance of models, when trained using the " +
          "already annotated data." +
          "</div>" +

          "</div>"
      },
      {
        title: "Export data",
        subtitle: "The fruits of your labor",
        text:
          "<div style='display: flex; flex-direction: row; justify-content: space-between; align-items: start; gap: 2em 4em; flex-wrap: wrap'> " +
          "<div style='width: 500px; flex-grow: 1;'>" +
          "Once you finish with all the annotation rounds (or your annotators get tired, whichever comes first), you are " +
          "ready to harvest the fruit of your (well, the annotators') labor! To export the dataset, complete with all " +
          "the annotations, start by going to the Export page within a project's domain. Apart from choosing the desired " +
          "file format of the exported dataset, you are also provided with the options to get aggregated and unlabeled " +
          "data. Don't worry - we'll get to that in a moment. It's important to note that you can export the dataset at " +
          "any point during a project's lifecycle. If this is " +
          "your first time exporting, you will have to press the Create button, but after that, you will always be able to " +
          "download the last generated version of the dataset." +
          "</div>" +

          "<div style='width: 500px; flex-grow: 1;'>" +
          "If data is not exported in aggregated format the fields included in the exported dataset are " +
          "the same ones that are allowed to be imported (<code style='color: black;'>document_id</code>, " +
          "<code style='color: black;'>text</code>, <code style='color: black;'>label</code> and " +
          "<code style='color: black;'>annotator</code>) plus the information if document was annotated using guided learning. " +
          "If you keep the 'Get aggregated data' option checked (recommended), we will aggregate the annotations of each " +
          "document to form a final annotation as well as provide you with label statistics. Dataset rows are:" +
          "<code style='color: black;'>document_id</code>, <code style='color: black;'>text</code>, row for each label named after" +
          "the label indicating percentage of annotations using that label, <code style='color: black;'>aggregated_label</code>" +
          "(the final aggregated label made considering all annotations), " +
          "<code style='color: black;'>num_labels</code> (the total number of labels used for the document), " +
          "<code style='color: black;'>num_annotators</code> (the total number of annotators that annotated " +
          "the document), <code style='color: black;'>round</code> (number of round in which document was distributed), " +
          "and flags <code style='color: black;'>AL</code> and <code style='color: black;'>GL</code> informing" +
          "if document was selected via active or guided learning. If the project is multi-label, the field " +
          "<code style='color: black;'>MASI_similarity</code> is used to inform about MASI similarity of the document's " +
          "annotations." +
          "</div>" +

          "<div style='width: 500px; flex-grow: 1;'>" +
          "When exporting as " +
          "a CSV file, the header will consist of these fields, and the rows will contain document data, with the text " +
          "returned exactly as you submitted it. With JSON, you will get a list of objects, where each object has the " +
          "four fields as properties, with values identical as in the CSV. " +
          "By default, the 'Get unlabeled data' option is unchecked - selecting it " +
          "will include documents which were still not annotated, with a blank string as their value." +
          "</div>" +

          "</div>"
      },
    ]
  }
});

