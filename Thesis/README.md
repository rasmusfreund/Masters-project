# LaTeX template for assignments at Aarhus University

This template has been created as a general template for various assignments during my master's in Bioinformatics at Aarhus University. Anyone should feel free to use it for whatever they may need, and change anything that doesn't suit their needs. Below I've attempted to create a general guide for using the template; I personally use [TeXstudio](https://www.texstudio.org/#download) to edit the source code, along with the [MiKTeX](https://miktex.org/download) distribution of LaTeX (if Mac users experience problems with MiKTeX, they might want to use [MacTeX](https://www.tug.org/mactex/mactex-download.html) instead).

The font used for the title and headers is the AU Passata font created by Aarhus University. The font can be downloaded for free [here](https://medarbejdere.au.dk/en/administration/communication/guidelines/guidelinesforfonts) - specifically, the AU Passata Regular and AU Passata Bold are being used. The main text is the Georgia font, which is standard for both Windows and Mac.

**Important note**: to compile the template "as is", it must be compiled using either LuaLaTeX or XeLaTeX due to the package *fontspec*, which is used to render the non-standard AU Passata font.



### Changing the front page
There are currently two versions of the template, the difference being the front page:

| | |
:-------------------------:|:-------------------------:
<img src='/src/fp_blue.png' alt='Blue front page' width='400'> | <img src='/src/fp_white.png' alt='White front page' width='400'>


The blue front page is enabled by having the following line uncommented:
```latex
% ------ Blue front page ------
\frontpageBlue{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{img/ausegl_hvid.png}{img/au_white.png}{}


% ------ White front page ------
%\frontpageWhite{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{img/ausegl.png}{img/au_blue.png}{}
```

and vice versa to get the white front page:
```latex
% ------ Blue front page ------
%\frontpageBlue{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{img/ausegl_hvid.png}{img/au_white.png}{}


% ------ White front page ------
\frontpageWhite{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{img/ausegl.png}{img/au_blue.png}{}
```

### Changing title, author, and images
The front pages can take up to eight arguments:

1. Title of the paper
2. Author number one
3. Department / affiliation of author number one
4. Author number two
5. Department / affiliation of author number two
6. Additional information
7. Top image (standard is the seal of Aarhus University)
8. Bottom image (standards is the logo of Aarhus University)

<img src='/src/fp_args.png' alt='Front page with arguments' width='400'>

To generate the front page as in the image above, the `\frontpageWhite` command must look like this:

```latex
\frontpageWhite{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{Sample additional info}{img/ausegl.png}{img/au_blue.png}

```
If you want to remove any specific element, simply delete the text within the curly brackets `{}`, but leave the brackets in the command.

Example: if there's only one author and the "additional info" text is not needed, the command would look like this:

```latex
\frontpageWhite{A systematic review of...}{Sample Author One}{Sample Department One}{}{}{}{img/ausegl.png}{img/au_blue.png}
```

This would produce the following front page:

<img src='/src/fp_oneauthor.png' alt='Front page with one author' width='400'>

Images on the front page can be changed if you'd rather want something else that relates to your work. As mentioned earlier, the larger [AU seal](https://medarbejdere.au.dk/en/administration/communication/guidelines/seal)
 is argument number seven in the front page command, and the smaller [AU logo](https://medarbejdere.au.dk/en/administration/communication/guidelines/guidelinesforlogo) is the final and eighth argument. For easy swapping of either, copy the image you want to use into the `img` directory that comes with the repository, and simply change the name in the front page command:

 ```latex
 \frontpageWhite{A systematic review of...}{Sample Author One}{Sample Department One}{Sample Author Two}{Sample Department Two}{Sample additional info}{img/MyOwnLargeImage.jpeg}{img/MyOwnSmallImage.png}
 ```

Note that depending on the size of the image you want to use, you may have to play around with the spacing on the front page. The following code snippet shows which lines to change on the white front page; the options are identical in the code for the blue version:

```latex
\newcommand{\frontpageWhite}[8]{
	\begin{titlepage}
		\centering
		\includegraphics[width=0.5\textwidth]{#7}\par
		\vspace{1cm} % <--- customize the space between the top image and the title
		{\fontsize{34}{40}\selectfont\color{sectioncolor}\titlefont #1\par}
		\vspace{2cm} % <--- customize the space between title and author one
		{\Large\headingfont\color{sectioncolor} #2\par}
		{\large\headingfont\color{sectioncolor} #3\par}
		\vspace{1cm} %<--- customize the space between the authors
		{\Large\headingfont\color{sectioncolor} #4\par}
		{\large\headingfont\color{sectioncolor} #5\par}
		\vspace{1cm} % <--- customize the space between the second author and the small image
		\includegraphics[width=0.5\textwidth]{#8}
		\vfill
		{\Large\headingfont\color{sectioncolor} #6}
	\end{titlepage}
}
```

### Headers

There's currently four different headers to choose between, which can be chosen by un-commenting the one of your choice:

#### Header with no department
```latex
% Configure the header
\pagestyle{fancy}
\fancyhf{} % Clears all header and footer fields
\fancyhead[L]{\includegraphics[height=1.1cm]{img/au_blue2.png}} % No department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/inano.png}} % Interdisciplinary Nanoscience department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/mbg.png}} % Molecular Biology department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/birc.png}} % Bioinformatics Research department
```
<img src='/src/nodep_header.png' alt='Header with no department' width='500'>

#### iNano header
```latex
% Configure the header
\pagestyle{fancy}
\fancyhf{} % Clears all header and footer fields
%\fancyhead[L]{\includegraphics[height=1.1cm]{img/au_blue2.png}} % No department
\fancyhead[L]{\includegraphics[height=1.5cm]{img/inano.png}} % Interdisciplinary Nanoscience department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/mbg.png}} % Molecular Biology department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/birc.png}} % Bioinformatics Research department
```
<img src='/src/inano_header.png' alt='iNano header' width='500'>

#### Molecular Biology header
```latex
% Configure the header
\pagestyle{fancy}
\fancyhf{} % Clears all header and footer fields
%\fancyhead[L]{\includegraphics[height=1.1cm]{img/au_blue2.png}} % No department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/inano.png}} % Interdisciplinary Nanoscience department
\fancyhead[L]{\includegraphics[height=1.5cm]{img/mbg.png}} % Molecular Biology department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/birc.png}} % Bioinformatics Research department
```
<img src='/src/molbio_header.png' alt='Molecular biology header' width='500'>

#### Bioinformatics Research Center header
```latex
% Configure the header
\pagestyle{fancy}
\fancyhf{} % Clears all header and footer fields
%\fancyhead[L]{\includegraphics[height=1.1cm]{img/au_blue2.png}} % No department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/inano.png}} % Interdisciplinary Nanoscience department
%\fancyhead[L]{\includegraphics[height=1.5cm]{img/mbg.png}} % Molecular Biology department
\fancyhead[L]{\includegraphics[height=1.5cm]{img/birc.png}} % Bioinformatics Research department
```
<img src='/src/birc_header.png' alt='BiRC header' width='500'>

### Footers
There are two different type of footers that changes the type of page numbering used. In the same way as with the headers, you can choose between them by un-commenting the relevant variant:

#### Current page only
```latex
\fancyfoot[C]{\headingfont\thepage} % Page number in footer
%\fancyfoot[C]{\headingfont\thepage\ of \pageref{LastPage}} % "m of n" page numbering in footer
```
<img src='/src/current_footer.png' alt='Current page footer' width='80'>

#### m of n pages
```latex
%\fancyfoot[C]{\headingfont\thepage} % Page number in footer
\fancyfoot[C]{\headingfont\thepage\ of \pageref{LastPage}} % "m of n" page numbering in footer
```
<img src='/src/m_of_n_footer.png' alt='m-of-n footer' width='100'>
