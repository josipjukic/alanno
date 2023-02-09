
![alanno_logo](https://user-images.githubusercontent.com/53186339/182347430-9d97ba90-adf2-4d3e-beae-ec72dbbf8d1b.png)

Alanno - Active Learning Annotation
===================================
<br>
Annotation tool powered by Active Learning.
A short demonstration video is available at https://www.youtube.com/watch?v=hPcHPM8ttvE.
<br>

# Usage

The application can be run in two modes:

- Local
- Production

You will need [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/). 

First, you need to clone the repository:

```bash
$ git clone https://github.com/josipjukic/alanno.git
$ cd alanno
```

### Local:
```bash
$ docker-compose -f docker-compose.local.yml up
```

### Production:
```bash
$ docker-compose -f docker-compose.prod.yml up
```

Note: An example of relevant environmental variables is given in the `.env` file. You should replace the placeholder values.

Go to <http://localhost:8000/> to access the application.


# Alanno Guide

## 1. Create a project

#### You need to get some work

As an administrator, you get to create new projects and be their manager. Creating a project is the necessary first step before any work is done. To create a project, fill in the project creation form on the `projects` page (you can always reach this page via the links in the sidebar). Once you name and describe your project, you'll need to declare its type and language. The type refers to the NLP problem for which we are annotating training data, while the language refers to the language of the documents in the dataset, and not the UI, unfortunately.

Scratch below the surface, and into the whole point of this application, by toggling the **Active Learning** switch on the form, which will display a whole new set of settings for your project. Every Active Learning method relies on an AI model to query the dataset in some way. In other words, if you want to use Active Learning, you will need to define the supporting NLP model. These settings are more technical in nature and require you to know a bit about NLP and Active Learning. Don't worry, though: you can change them anytime on the project Settings page.

As a regular user, and by this we mean annotator, your only option is to be invited to join a project by a project's moderator. They will do this by sending you an access code, which you can then safely type into the appropriate form on the bottom of the `Projects` page. Once you click **Join**, you are officially a member of the project, and soon enough you will be able to start annotating documents.


## 2. Add members

#### We don't want it to be lonely here

Going back to the `Projects` page, we now see that apart from the forms for creating and joining a project, it also shows all the projects you're a member of. On the surface, clicking on a project just takes you to its `Settings` page. But actually, you are now in the project's "domain" - everything you do is related to this particular project, until you eventually go back to the `Projects` page, or the homepage. You can verify this by opening the sidebar - it will show several new, project-related options (starting with **Settings**).

Let's stay on the Settings page for a while, since we're here. The first, and most important part of this page is the project access code. You need this to invite other people to join your project as annotators by sending them the access code. They will have to create an account if they don't have it already, and then paste the access code into the project joining form. After you import a dataset, you will be able to distribute documents for annotation to all the members you added.

Apart from the access code, the `Settings` page provides all the project information, as well as the ability to help your annotators by adding project guidelines. If you enabled Active Learning for your project, the project information is followed by, as promised, a section where you can change all the related settings. Finally, there is an option to delete the project, in case something goes wrong with it. But please come back!


## 3. Define labels

#### But only if you feel like it

If a part of your dataset has already been labeled, it might make sense to define the set of labels before importing the dataset. That way the labels will be recognized and stored when the dataset is imported. This is especially useful when Active Learning is used, because then you can use these documents to enable a **"warm start"** for the supporting model, increasing the efficiency of the whole process.

Label creation itself is fairly straightforward - you can choose the label name, its shortcut key (the annotation process can be done using just the keyboard, which makes things a bit faster), and its background and text colors. Next to the label creation form you will see a list of all the labels you defined for the project so far. These are the options which will be given to annotators when they annotate documents.


## 4. Import a dataset

#### Finally something to work with

In order to annotate documents, we first need to have some documents to annotate. To import a dataset, open the Import page within the project's domain. Some basic instructions are written there as well, but it's certainly worth repeating:  

*   You can only import CSV files
*   The CSV file must contain a header row
*   Supported headers are: `text`, `document_id`, `label` and `annotator`
*   Only the `text` column is required
*   The text may include line breaks `\n` and/or HTML tags, which will be used to properly format the text during annotation

We know that you may already have a partially annotated dataset, so we made sure you don't lose those annotations. You can simply add the `annotator` and `label` headers, and fill in the values for all pre-annotated documents. There are several constraints and conditions, though:

*   If the label isn't present or isn't defined in the project, the annotation is ignored
*   If the annotator doesn't correspond to a user's username, the label will be added as an anonymous annotation
*   Anonymous annotations are used to enable a warm start for the supporting model for Active Learning
*   Currently, importing annotations is only supported for classification projects

The `document_id` column should contain a numerical identifier of the document, used if the same document appears multiple times in the dataset. This can happen in a single-label setup (multiple users annotated a document) and in a multi-label setup (a single user gave multiple labels to a document). With a document ID, we can recognize this and correctly store all the annotations. If your dataset is not pre-annotated or doesn't contain multiple rows for a single document, you don't really need this column.


## 5. Distribute documents

#### Sharing is caring

You can verify that your data has been successfully uploaded by visiting the project's Data page. Now is the time to prepare the stage for the work to come, i.e. the annotating process. The annotators are not independent of you, they can't just open any document they want and annotate it - that would defeat the purpose of Active Learning. Instead, the system decides which documents will get distributed, and you decide when and to whom. In Alanno, documents are distributed in rounds, where each round is independent of others. This way, you can adjust the workload for each round, depending on the availability of your annotators.

Rounds (also called batches) for a project are generated on the project's Control page. When you visit the Control page, you will be greeted with a form where you can specify the settings of the round to be generated, such as the annotators included in the round, the number of annotators per document, and, of course, the number of documents to be annotated in this round. In case Active Learning is enabled, you also get to define additional settings, such as the proportion of random data to be annotated (data **not selected** by an AL strategy), the proportion of test data, and the possibility to enable warm-starting the supporting model.


## 6. Annotate data

#### We humans sure like to stick labels

Regardless of your role on a project, once the project manager has distributed documents to you, you can start annotating them. If you're the manager, you will need to open the Annotation page within the project's domain. Otherwise, annotating is the only thing you can do, so it suffices to just enter the project's domain. The annotation process is the core of Alanno, and we tried to make it as user-friendly as possible.

On the left side of the page you will see your progress in this round, followed by a list of documents that were distributed to you. You can filter these documents by their completion status, as well as search them for keywords. The bottom of the page contains a lock button sandwiched between navigation buttons. The former is used to submit your annotation, while the latter are used to quickly go from one document to the next (or previous). As long as the round is active (i.e. until the manager creates a new round), you are free to unlock, change and lock the annotations you provided.

The center of this page will always display the document itself, with any formatting that was provided as HTML tags during dataset import. However, the actual manner of annotation depends on the project type and other modifiers. This will always be explained on the annotation page itself, but just in case: classification projects require you to select one (single-label) or more (multi-label) labels shown below the document, while sequence labeling projects require you to select a part of text **and** a label to apply to the selected text.


## 7. Track annotation progress

#### Make sure there are no slackers

It is very important to know how your project is doing, and more precisely, how the annotators are managing with the workload. To this end, we created a Stats page within the project's domain where you can monitor various kinds of metrics related to your project, together with pretty visualisations. The statistics are updated periodically, in case a change has been detected. However, we know you may want the data sooner, so we enabled you to force a statistics recalculation. This is especially useful when you think the statistics are lagging behind the real situation. It is important to note that the statistics are round-sensitive, meaning the data and graphs you see are referring only to the specific round you selected at the top of the page. Alternatively, you can also select 'All rounds' to show statistics of the project's whole lifecycle.

The provided statistics are somewhat grouped into cards with common semantics. The Annotation Progress card displays the current annotator progress in the selected round. It shows an overall progress, followed by an individual progress indicator, for each annotator on the project. Following this is the Distribution progress card, displaying the number of documents that have already been distributed. This is a kind of a 'global progress', since the project is definitely finished once all the documents have been distributed and annotated. Next to this is the Label frequency card, showing a histogram of label frequencies in the selected round. Finally, since Alanno is mostly used in a multi-annotator setup, it is necessary to provide insight into the Inter Annotator Agreement. We display a visualisation of both pairwise and average agreement per user, using Cohen's kappa for single-label setups, and Fleiss' kappa for multi-label ones. If Active Learning is enabled, after a few rounds you will also see graphs showing the train and test performance of models, when trained using the already annotated data.


## 8. Export data

#### The fruits of your labor

Once you finish with all the annotation rounds (or your annotators get tired, whichever comes first), you are ready to harvest the fruit of your (well, the annotators') labor! To export the dataset, complete with all the annotations, start by going to the Export page within a project's domain. Apart from choosing the desired file format of the exported dataset, you are also provided with the options to get aggregated and unlabeled data. Don't worry - we'll get to that in a moment. It's important to note that you can export the dataset at any point during a project's lifecycle. If this is your first time exporting, you will have to press the Create button, but after that, you will always be able to download the last generated version of the dataset.

By default, the fields included in the exported dataset are the same ones that are allowed to be imported: `document_id`, `text`, `label` and `annotator`. When exporting as a CSV file, the header will consist of these fields, and the rows will contain document data, with the text returned exactly as you submitted it. With JSON, you will get a list of objects, where each object has the four fields as properties, with values identical as in the CSV. If you keep the 'Get aggregated data' option checked (recommended), we will aggregate the annotations of each document to form a final annotation as well as provide you with label statistics. By default, the 'Get unlabeled data' option is unchecked - selecting it will include documents which were still not annotated, with a blank string as their value.
