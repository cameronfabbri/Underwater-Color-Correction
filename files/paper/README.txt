
December 27, 2012


IEEEtran is a LaTeX class for authors of the Institute of Electrical and
Electronics Engineers (IEEE) transactions journals and conferences.
The latest version of the IEEEtran package can be found at CTAN:

http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/

as well as within IEEE's site:

http://www.ieee.org/

For latest news, helpful tips, answers to frequently asked questions,
beta releases and other support, visit the IEEEtran home page at my
website:

http://www.michaelshell.org/tex/ieeetran/

V1.8 is a significant update over the 1.7a release. For a full list of
changes, please read the file changelog.txt. The most notable changes
include:


 1) New transmag class option to support the IEEE Transactions on Magnetics
    format. Thanks to Wei Yingkang, Sangmin Suh and Benjamin Gaussens
    for suggestions and beta testing.

 2) The \IEEEcompsoctitleabstractindextext and 
    \IEEEdisplaynotcompsoctitleabstractindextext
    commands have been deprecated in favor of their
    \IEEEtitleabstractindextext and \IEEEdisplaynontitleabstractindextext
    (observe that the "not" has changed to "non") equivalents. This change
    generalizes and decouples them from compsoc mode because the new
    transmag mode also uses them now.

 3) Added new *-forms of \IEEEyesnumber*, \IEEEnonumber*, \IEEEyessubnumber*,
    and \IEEEnosubnumber* (the non-star form of the latter is also new) which
    persist across IEEEeqnarray lines until countermanded. To provide for
    continued subequations across instances of IEEEeqnarrays as well as for
    subequations that follow a main equation (e.g., 14, 14a, 14b ...)
    \IEEEyessubnumber no longer automatically increments the equation number
    on it's first invocation of a subequation group. Invoke both
    \IEEEyesnumber\IEEEyessubnumber together to start a new
    equation/subequation group.
 
 4) Hyperref links now work with IEEEeqnarray equations.
    Thanks to Stefan M. Moser for reporting this problem.

 5) Revised spacing at top of top figures and tables to better
    align with the top main text lines as IEEE does in its journals. 
    Thanks to Dirk Beyer for reporting this issue and beta testing.


Best wishes for all your publication endeavors,

Michael Shell
http://www.michaelshell.org/


********************************** Files **********************************

README                 - This file.

IEEEtran.cls           - The IEEEtran LaTeX class file.

changelog.txt          - The revision history.

IEEEtran_HOWTO.pdf     - The IEEEtran LaTeX class user manual.

bare_conf.tex          - A bare bones starter file for conference papers.

bare_jrnl.tex          - A bare bones starter file for journal papers.

bare_jrnl_compsoc.tex  - A bare bones starter file for Computer Society
                         journal papers.

bare_jrnl_transmag.tex - A bare bones starter file for IEEE Transactions
                         on Magnetics journal papers.

bare_adv.tex           - A bare bones starter file showing advanced
                         techniques such as conditional compilation,
                         hyperlinks, PDF thumbnails, etc. The illustrated
                         format is for a Computer Society journal paper.

***************************************************************************
Legal Notice:
This code is offered as-is without any warranty either expressed or
implied; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE! 
User assumes all risk.
In no event shall IEEE or any contributor to this code be liable for
any damages or losses, including, but not limited to, incidental,
consequential, or any other damages, resulting from the use or misuse
of any information contained here.

All comments are the opinions of their respective authors and are not
necessarily endorsed by the IEEE.

This work is distributed under the LaTeX Project Public License (LPPL)
( http://www.latex-project.org/ ) version 1.3, and may be freely used,
distributed and modified. A copy of the LPPL, version 1.3, is included
in the base LaTeX documentation of all distributions of LaTeX released
2003/12/01 or later.
Retain all contribution notices and credits.
** Modified files should be clearly indicated as such, including  **
** renaming them and changing author support contact information. **

File list of work: IEEEtran.cls, IEEEtran_HOWTO.pdf, bare_adv.tex,
                   bare_conf.tex, bare_jrnl.tex, bare_jrnl_compsoc.tex,
                   bare_jrnl_transmag.tex
***************************************************************************
