---
title: Blogging with GitHub Pages and Jekyll
author: Andre Larsson
date: 2021-02-19
key: bloggpj21
comment: true
tags: github jekyll blog github-pages
output:
 md_document:
  pandoc_args: ["--wrap=none"]
  variant: gfm
  toc: true
  preserve_yaml: TRUE
---

# Introduction

This post will quickly go over how this website was created on Github
pages with Jekyll. It will assume you are already familiar with
[git](https://git-scm.com/).

## Motivation

This post was written to explain the process of getting a blog up and
running, specifically using Jekyll, GitHub Pages developing in
Linux/Ubuntu. I hope it could serve as an inspiration for anyone
wanting to create their own. It is aimed at someone who already know
git and want a simple tool for blogging, my aim being to write a
streamlined set of steps to get from a github account to a Jekyll
blog, understanding basic configuration and adding posts, in the
shortest amount of time.

## What is GitHub pages?

GitHub Pages is a service provided by GitHub, where you can host a
website such as a blog, where all files are saved in a repository with
the special name *myusername.github.io*, which will give you a website
available at https://myusername.github.io. It provides integration
with Jekyll, as it will automatically build a new site whenever files
in the repository are updated or added.

## What is Jekyll?

Jekyll is a static site generator written in Ruby by one of founders
of GitHub. Instead of writing everything from scratch, Jekyll will
build a site based on how its configured, and after its set up you can
simply add new posts in e.g. the simple markdown format (see
https://www.markdownguide.org/), with Jekyll taking care of
the rest (layout, navigation, consistency).

## Alternatives

### Gitlab pages

You could also host your site on Gitlab, which seems to have better support
for other static site generators

https://about.gitlab.com/stages-devops-lifecycle/pages/

### Static site generators

There are several other tools for generating static websites, for example

- [Hugo](https://gohugo.io/)
- [Middleman](https://middlemanapp.com/)
- [Eleventy](https://www.11ty.dev/)

I choose Jekyll/Github since it was relatively easy to setup and there
seemed to be good integration between these two, for basic blogging
purposes.

### Blog platforms

There are other platforms or software for blogging such as

- [Wordpress](https://wordpress.org/)
- [Wix](https://www.wix.com/)
- [Medium](https://medium.com/)

and many more...

# Setting up GitHub Pages

Assuming you already have a [github](https://github.com/) account and
is familiar with git, setting up GitHub Pages is very easy.

As mentioned on the [official guide for github pages](https://pages.github.com/), all you need to do is create a repository with the
special name *myusername.github.io*. Then, if you e.g. add a simple
"index.html" file to the repository it shown to anyone visiting the
adress *https://myusername.github.io*. Simple enough.

Though, we want to do better than a single html page, which is why
we will use the static site generator Jekyll.

Therefore, we do not create this special repository just yet, instead we
first install Jekyll, then, find a theme we like and fork it to this
special repository.

# Installing Jekyll

Jekyll can be installed following the instructions on [jekyllrb.com/docs/installation](https://jekyllrb.com/docs/installation/). I followed the [guide for Ubuntu](https://jekyllrb.com/docs/installation/ubuntu/) linked from this page which worked well for me.

Overall, I found the documentation for Jekyll to be well-written and
to-the-point. Even though we will fork a complete theme, I recommend
going through the [Step by Step Tutorial](https://jekyllrb.com/docs/step-by-step/01-setup/) to better
understand how Jekyll operates and how you can customise your site.

# Start from a jekyll theme

## Find a theme

There are many Jekyll themes available at
[jekyllthemes.org](http://jekyllthemes.org/). This site is based on
the [TeXt Theme](https://github.com/kitian616/jekyll-TeXt-theme) which
had many features such as comments, further customization and a nice
interface.

## Fork the theme

Once you found a theme you like, you could either clone it, or, as will
be in this tutorial,
[fork](https://guides.github.com/activities/forking/) it to a repository called *myusername.github.io*, e.g., using the fork button on the
website for github repository or the command-line.

The benefit of forking (compared to cloning) is that you can modify
and add content to your site while still keeping the connection to the
original template, keeping track of any major updates.

After forking the theme to the special github repository of now, the
default theme should be made available at *https://myusername.github.io* (it might take
a couple of moments for it to build).

## Set up your development environment

Next we need way to develop our site locally and preview our
changes. What we want to do is clone our repository, and set up Jekyll
to generate the site and preview it (locally) in our browser as we
make changes to it.

For the TeXt theme, which this site is based on, there is a
[quick-start](https://tianqi.name/jekyll-TeXt-theme/docs/en/quick-start)
guide, please check for any documentation of the theme you
dowloaded for more detailed instructions on how to configure it. If
there is no documentation, check the [docs for Jekyll](https://jekyllrb.com/docs/), as they will
hopefully provide enough information.

Following the instructions for TeXt theme, we need to install all
dependencies (called gems for Ruby), which can be done by bundle using
the following command (Ubuntu), this have to be done if your theme has any dependencies listed
```
bundle install --path vendor/bundle
```
Assuming the above command succesfully completes, you can now create the website with Jekyll and preview it with
```
bundle exec jekyll serve
```
This should print an IP-adress which you can copy-paste in your browser to access your site. Now we have a
simple development environment set up, where you can make changes to your local copy, preview them, and push
them to GitHub when happy.

## Configuring

At this point we should have a site up-and-running at
https://myusername.github.io, and a local clone which we now can
customise to our hearts content, to make it our own.

Likely, at a minimum you want to change the file *_config.yml* for site
description, author, links, etc., *about.md* for the about page.

Refer to the [Jekyll docs](https://jekyllrb.com/docs/) for more
information of what the different files do and how to better configure
your site. For example, the folder *_site* is the actual site
automatically generated by Jekyll, in all its glory, while the folder
*_includes* contains files that can be reused across the
entire site.

## Adding blog posts

To write a blog post, create a file with the name
*YEAR-MONTH-DAY-title.MARKUP* in the folder *_posts*. After building the site, it should automatically be
added to your website. See more at [jekyllrb.com/docs/posts/](https://jekyllrb.com/docs/posts/),
and take note of changing the
[front matter](https://jekyllrb.com/docs/front-matter/) to set the
title, layout, author of the post. In essence, you need to check the
documentation/source code for the Jekyll theme you are using to
know what you want to specify here.

The blog posts can be written in markdown, there are many handy guides
for writing in markdown available online, one is this [cheat sheet](https://www.markdownguide.org/cheat-sheet).

When you are happy with your changes, you can push them to GitHub,
which will rebuild your site if necessary.

# Conclusion

At the end of this tutorial you will have learned to create a simple
blog using GitHub and Jekyll. I thought this would be a very
simple process, and in some regard it was, though it took a some time
to put it all together and reading up on how to use Jekyll. Once set
up, it is easy to add content (posts) and GitHub conveniently will
build the site for you as long as you stick to using
Jekyll.

All-in-all Jekyll/GitHub seems to work very well as for creating a
smaller website or blog, it is what I am using here after all, and I recommend checking it out.
