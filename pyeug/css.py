scrollbar_css = """
::-webkit-scrollbar {
  width: 5px;
}
::-webkit-scrollbar-track {
  background: white;
}
::-webkit-scrollbar-thumb {
  background: #888;
}
::-webkit-scrollbar-thumb:hover {
  background: #555;
}
"""

multichoice_css = """
.choices__item.choices__item--selectable[aria-selected='true'] {
    background-color: #6E808C !important;
    color: white !important; /* Change text color to white for better readability */
}
"""

switch_css = """
/* Inside component's internal style */
.body .bar {
    background-color: #CCCCCC !important;
}
.body .knob {
    background-color: #999999 !important;
}
"""


