// Copied from .../google3/javascript/common.js

//------------------------------------------------------------------------
// This file contains common utilities and basic javascript infrastructure.
//
// Notes:
// * Press 'D' to toggle debug mode.
//
// Functions:
//
// - Assertions
// DEPRECATED: Use assert.js
// AssertTrue(): assert an expression. Throws an exception if false.
// Fail(): Throws an exception. (Mark block of code that should be unreachable)
// AssertEquals(): assert that two values are equal.
// AssertNumArgs(): assert number of arguments for the function
// AssertType(): assert that a value has a particular type
//
// - Cookies
// SetCookie(): Sets a cookie.
// GetCookie(): Gets a cookie value.
//
// - Dynamic HTML/DOM utilities
// MaybeGetElement(): get an element by its id
// GetElement(): get an element by its id
// ShowElement(): Show/hide element by setting the "display" css property.
// ShowBlockElement(): Show/hide block element
// AppendNewElement(): Create and append a html element to a parent node.
// HasClass(): check if element has a given class
// AddClass(): add a class to an element
// RemoveClass(): remove a class from an element
//
// - Window/Screen utiltiies
// GetPageOffsetLeft(): get the X page offset of an element
// GetPageOffsetTop(): get the Y page offset of an element
// GetPageOffset(): get the X and Y page offsets of an element
// GetPageOffsetRight() : get X page offset of the right side of an element
// GetPageOffsetBottom() : get Y page offset of the bottom of an element
// GetScrollTop(): get the vertical scrolling pos of a window.
// GetScrollLeft(): get the horizontal scrolling pos of a window
//
// - String utilties
// HtmlEscape(): html escapes a string
// HtmlUnescape(): remove html-escaping.
// CollapseWhitespace(): collapse multiple whitespace into one whitespace.
// Trim(): trim whitespace on ends of string
// IsEmpty(): check if CollapseWhiteSpace(String) == ""
// IsLetterOrDigit(): check if a character is a letter or a digit
//
// - TextArea utilities
// SetCursorPos(): sets the cursor position in a textfield
//
// - Array utilities
// FindInArray(): do a linear search to find an element value.
// DeleteArrayElement(): return a new array with a specific value removed.
//
// - Miscellaneous
// IsDefined(): returns true if argument is not undefined
//------------------------------------------------------------------------

// browser detection
var agent = navigator.userAgent.toLowerCase();
var is_ie = (agent.indexOf('msie') != -1);
//var is_ie5 = (agent.indexOf('msie 5') != -1 && document.all);
var is_konqueror = (agent.indexOf('konqueror') != -1);
var is_safari = (agent.indexOf('safari') != -1) || is_konqueror;
var is_nav = !is_ie && !is_safari && (agent.indexOf('mozilla') != -1);
var is_win = (agent.indexOf('win') != -1);
delete agent;


var BACKSPACE_KEYCODE = 8;
var COMMA_KEYCODE = 188;                // ',' key
var DEBUG_KEYCODE = 68;                 // 'D' key
var DELETE_KEYCODE = 46;
var DOWN_KEYCODE = 40;                  // DOWN arrow key
var ENTER_KEYCODE = 13;                 // ENTER key
var ESC_KEYCODE = 27;                   // ESC key
var LEFT_KEYCODE = 37;                  // LEFT arrow key
var RIGHT_KEYCODE = 39;                 // RIGHT arrow key
var SPACE_KEYCODE = 32;                 // space bar
var TAB_KEYCODE = 9;                    // TAB key
var UP_KEYCODE = 38;                    // UP arrow key
var SHIFT_KEYCODE = 16;

//------------------------------------------------------------------------
// Assertions
// DEPRECATED: Use assert.js
//------------------------------------------------------------------------
/**
 * DEPRECATED: Use assert.js
 */
function raise(msg) {
  if (typeof Error != 'undefined') {
    throw new Error(msg || 'Assertion Failed');
  } else {
    throw (msg);
  }
}

/**
 * DEPRECATED: Use assert.js
 *
 * Fail() is useful for marking logic paths that should
 * not be reached. For example, if you have a class that uses
 * ints for enums:
 *
 * MyClass.ENUM_FOO = 1;
 * MyClass.ENUM_BAR = 2;
 * MyClass.ENUM_BAZ = 3;
 *
 * And a switch statement elsewhere in your code that
 * has cases for each of these enums, then you can
 * "protect" your code as follows:
 *
 * switch(type) {
 *   case MyClass.ENUM_FOO: doFooThing(); break;
 *   case MyClass.ENUM_BAR: doBarThing(); break;
 *   case MyClass.ENUM_BAZ: doBazThing(); break;
 *   default:
 *     Fail("No enum in MyClass with value: " + type);
 * }
 *
 * This way, if someone introduces a new value for this enum
 * without noticing this switch statement, then the code will
 * fail if the logic allows it to reach the switch with the
 * new value, alerting the developer that he should add a
 * case to the switch to handle the new value he has introduced.
 *
 * @param {string} opt_msg to display for failure
 *                 DEFAULT: "Assertion failed"
 */
function Fail(opt_msg) {
  if (opt_msg === undefined) opt_msg = 'Assertion failed';
  if (IsDefined(DumpError)) DumpError(opt_msg + '\n');
  raise(opt_msg);
}

/**
 * DEPRECATED: Use assert.js
 *
 * Asserts that an expression is true (non-zero and non-null).
 *
 * Note that it is critical not to pass logic
 * with side-effects as the expression for AssertTrue
 * because if the assertions are removed by the
 * JSCompiler, then the expression will be removed
 * as well, in which case the side-effects will
 * be lost. So instead of this:
 *
 *  AssertTrue( criticalComputation() );
 *
 * Do this:
 *
 *  var result = criticalComputation();
 *  AssertTrue(result);
 *
 * @param {anything} expression to evaluate
 * @param {string}   opt_msg to display if the assertion fails
 *
 */
function AssertTrue(expression, opt_msg) {
  if (!expression) {
    if (opt_msg === undefined) opt_msg = 'Assertion failed';
    Fail(opt_msg);
  }
}

/**
 * DEPRECATED: Use assert.js
 *
 * Asserts that two values are the same.
 *
 * @param {anything} val1
 * @param {anything} val2
 * @param {string} opt_msg to display if the assertion fails
 */
function AssertEquals(val1, val2, opt_msg) {
  if (val1 != val2) {
    if (opt_msg === undefined) {
      opt_msg = "AssertEquals failed: <" + val1 + "> != <" + val2 + ">";
    }
    Fail(opt_msg);
  }
}

/**
 * DEPRECATED: Use assert.js
 *
 * Asserts that a value is of the provided type.
 *
 *   AssertType(6, Number);
 *   AssertType("ijk", String);
 *   AssertType([], Array);
 *   AssertType({}, Object);
 *   AssertType(ICAL_Date.now(), ICAL_Date);
 *
 * @param {anything} value
 * @param {constructor function} type
 * @param {string} opt_msg to display if the assertion fails
 */
function AssertType(value, type, opt_msg) {
  // for backwards compatability only
  if (typeof value == type) return;

  if (value || value == "") {
    try {
      if (type == AssertTypeMap[typeof value] || value instanceof type) return;
    } catch (e) { /* failure, type was an illegal argument to instanceof */ }
  }
  if (opt_msg === undefined) {
    if (typeof type == 'function') {
      var match = type.toString().match(/^\s*function\s+([^\s\{]+)/);
      if (match) type = match[1];
    }
    opt_msg = "AssertType failed: <" + value + "> not typeof "+ type;
  }
  Fail(opt_msg);
}

var AssertTypeMap = {
  'string'  : String,
  'number'  : Number,
  'boolean' : Boolean
};

/**
 * DEPRECATED: Use assert.js
 *
 * Asserts that the number of arguments to a
 * function is num. For example:
 *
 * function myFunc(one, two, three) [
 *   AssertNumArgs(3);
 *   ...
 * }
 *
 * myFunc(1, 2); // assertion fails!
 *
 * Note that AssertNumArgs does not take the function
 * as an argument; it is simply used in the context
 * of the function.
 *
 * @param {int} number of arguments expected
 * @param {string} opt_msg to display if the assertion fails
 */
function AssertNumArgs(num, opt_msg) {
  var caller = AssertNumArgs.caller;  // This is not supported in safari 1.0
  if (caller && caller.arguments.length != num) {
    if (opt_msg === undefined) {
      opt_msg = caller.name + ' expected ' + num + ' arguments '
                  + ' but received ' + caller.arguments.length;
    }
    Fail(opt_msg);
  }
}

//------------------------------------------------------------------------
// Cookies
//------------------------------------------------------------------------
var ILLEGAL_COOKIE_CHARS_RE = /[\s;]/
/**
 * Sets a cookie.
 * The max_age can be -1 to set a session cookie. To expire cookies, use
 * ExpireCookie() instead.
 *
 * @param name The cookie name.
 * @param value The cookie value.
 * @param opt_max_age The max age in seconds (from now). Use -1 to set a
 *   session cookie. If not provided, the default is -1 (i.e. set a session
 *   cookie).
 * @param opt_path The path of the cookie, or null to not specify a path
 *   attribute (browser will use the full request path). If not provided, the
 *   default is '/' (i.e. path=/).
 * @param opt_domain The domain of the cookie, or null to not specify a domain
 *   attribute (brower will use the full request host name). If not provided,
 *   the default is null (i.e. let browser use full request host name).
 * @return Void.
 */
function SetCookie(name, value, opt_max_age, opt_path, opt_domain) {

  value = '' + value;
  AssertTrue((typeof name == 'string' &&
              typeof value == 'string' &&
              !name.match(ILLEGAL_COOKIE_CHARS_RE) &&
              !value.match(ILLEGAL_COOKIE_CHARS_RE)),
             'trying to set an invalid cookie');

  if (!IsDefined(opt_max_age)) opt_max_age = -1;
  if (!IsDefined(opt_path)) opt_path = '/';
  if (!IsDefined(opt_domain)) opt_domain = null;

  var domain_str = (opt_domain == null) ? '' : ';domain=' + opt_domain;
  var path_str = (opt_path == null) ? '' : ';path=' + opt_path;

  var expires_str;

  // Case 1: Set a session cookie.
  if (opt_max_age < 0) {
    expires_str = '';

  // Case 2: Expire the cookie.
  // Note: We don't tell people about this option in the function doc because
  // we prefer people to use ExpireCookie() to expire cookies.
  } else if (opt_max_age == 0) {
    // Note: Don't use Jan 1, 1970 for date because NS 4.76 will try to convert
    // it to local time, and if the local time is before Jan 1, 1970, then the
    // browser will ignore the Expires attribute altogether.
    var pastDate = new Date(1970, 1 /*Feb*/, 1);  // Feb 1, 1970
    expires_str = ';expires=' + pastDate.toUTCString();

  // Case 3: Set a persistent cookie.
  } else {
    var futureDate = new Date(Now() + opt_max_age * 1000);
    expires_str = ';expires=' + futureDate.toUTCString();
  }

  document.cookie = name + '=' + value + domain_str + path_str + expires_str;
}

/** Returns the value for the first cookie with the given name
 * @param name : string
 * @return a string or the empty string if no cookie found.
 */
function GetCookie(name) {
  var nameeq = name + "=";
  var cookie = String(document.cookie);
  for (var pos = -1; (pos = cookie.indexOf(nameeq, pos + 1)) >= 0;) {
    var i = pos;
    // walk back along string skipping whitespace and looking for a ; before
    // the name to make sure that we don't match cookies whose name contains
    // the given name as a suffix.
    while (--i >= 0) {
      var ch = cookie.charAt(i);
      if (ch == ';') {
        i = -1;  // indicate success
        break;
      } else if (' \t'.indexOf(ch) < 0) {
        break;
      }
    }
    if (-1 === i) {  // first cookie in the string or we found a ;
      var end = cookie.indexOf(';', pos);
      if (end < 0) { end = cookie.length; }
      return cookie.substring(pos + nameeq.length, end);
    }
  }
  return "";
}


//------------------------------------------------------------------------
// Time
//------------------------------------------------------------------------
function Now() {
  return (new Date()).getTime();
}

//------------------------------------------------------------------------
// Dynamic HTML/DOM utilities
//------------------------------------------------------------------------
// Gets a element by its id, may return null
function MaybeGetElement(win, id) {
  return win.document.getElementById(id);
}

// Same as MaybeGetElement except that it throws an exception if it's null
function GetElement(win, id) {
  var el = win.document.getElementById(id);
  if (!el) {
    DumpError("Element " + id + " not found.");
  }
  return el;
}

// Gets elements by its id/name
// IE treats getElementsByName as searching over ids, while Moz use names.
// so tags must have both id and name as the same string
function GetElements(win, id) {
  return win.document.getElementsByName(id);
}

// Show/hide an element.
function ShowElement(el, show) {
  el.style.display = show ? "" : "none";
}

// Show/hide a block element.
// ShowElement() doesn't work if object has an initial class with display:none
function ShowBlockElement(el, show) {
  el.style.display = show ? "block" : "none";
}

// Show/hide an inline element.
// ShowElement() doesn't work when an element starts off display:none.
function ShowInlineElement(el, show) {
  el.style.display = show ? "inline" : "none";
}

// Append a new HTML element to a HTML node.
function AppendNewElement(win, parent, tag) {
  var e = win.document.createElement(tag);
  parent.appendChild(e);
  return e;
}

// Create a new TR containing the given td's
function Tr(win, tds) {
  var tr = win.document.createElement("TR");
  for (var i = 0; i < tds.length; i++) {
    tr.appendChild(tds[i]);
  }
  return tr;
}

// Create a new TD, with an optional colspan
function Td(win, opt_colspan) {
  var td = win.document.createElement("TD");
  if (opt_colspan) {
    td.colSpan = opt_colspan;
  }
  return td;
}


// Check if an element has a given class
function HasClass(el, cl) {
  if (el == null || el.className == null) return false;
  var classes = el.className.split(" ");
  for (var i = 0; i < classes.length; i++) {
    if (classes[i] == cl) {
      return true;
    }
  }
  return false;
}

// Add a class to element
function AddClass(el, cl) {
  if (HasClass(el, cl)) return;
  el.className += " " + cl;
}

// Remove a class from an element
function RemoveClass(el, cl) {
  if (el.className == null) return;
  var classes = el.className.split(" ");
  var result = [];
  var changed = false;
  for (var i = 0; i < classes.length; i++) {
    if (classes[i] != cl) {
      if (classes[i]) { result.push(classes[i]); }
    } else {
      changed = true;
    }
  }
  if (changed) { el.className = result.join(" "); }
}

// Performs an in-order traversal of the tree rooted at the given node
// (excluding the root node) and returns an array of nodes that match the
// given selector. The selector must implement the method:
//
// boolean select(node);
//
// This method is a generalization of the DOM method "getElementsByTagName"
//
function GetElementsBySelector(root, selector) {
  var nodes = [];
  for (var child = root.firstChild; child; child = child.nextSibling) {
    AddElementBySelector_(child, selector, nodes);
  }
  return nodes;
}

// Recursive helper for GetElemnetsBySelector()
function AddElementBySelector_(root, selector, nodes) {
  // First test the parent
  if (selector.select(root)) {
    nodes.push(root);
  }

  // Then recurse through the children
  for (var child = root.firstChild; child; child = child.nextSibling) {
    AddElementBySelector_(child, selector, nodes);
  }
}

//------------------------------------------------------------------------
// Window/screen utilities
// TODO: these should be renamed (e.g. GetWindowWidth to GetWindowInnerWidth
// and moved to geom.js)
//------------------------------------------------------------------------
// Get page offset of an element
function GetPageOffsetLeft(el) {
  var x = el.offsetLeft;
  if (el.offsetParent != null)
    x += GetPageOffsetLeft(el.offsetParent);
  return x;
}

// Get page offset of an element
function GetPageOffsetTop(el) {
  var y = el.offsetTop;
  if (el.offsetParent != null)
    y += GetPageOffsetTop(el.offsetParent);
  return y;
}

// Get page offset of an element
function GetPageOffset(el) {
  var x = el.offsetLeft;
  var y = el.offsetTop;
  if (el.offsetParent != null) {
    var pos = GetPageOffset(el.offsetParent);
    x += pos.x;
    y += pos.y;
  }
  return {x: x, y: y};
}

function GetPageOffsetRight(el) {
  return GetPageOffsetLeft(el) + el.offsetWidth;
}

function GetPageOffsetBottom(el) {
  return GetPageOffsetTop(el) + el.offsetHeight;
}

// Get the y position scroll offset.
function GetScrollTop(win) {
  // all except Explorer
  if ("pageYOffset" in win) {
    return win.pageYOffset;
  }
  // Explorer 6 Strict Mode
  else if ("documentElement" in win.document &&
           "scrollTop" in win.document.documentElement) {
    return win.document.documentElement.scrollTop;
  }
  // other Explorers
  else if ("scrollTop" in win.document.body) {
    return win.document.body.scrollTop;
  }

  return 0;
}

// Get the x position scroll offset.
function GetScrollLeft(win) {
  // all except Explorer
  if ("pageXOffset" in win) {
    return win.pageXOffset;
  }
  // Explorer 6 Strict Mode
  else if ("documentElement" in win.document &&
           "scrollLeft" in win.document.documentElement) {
    return win.document.documentElement.scrollLeft;
  }
  // other Explorers
  else if ("scrollLeft" in win.document.body) {
    return win.document.body.scrollLeft;
  }

  return 0;
}

//------------------------------------------------------------------------
// String utilities
//------------------------------------------------------------------------
// Do html escaping
var amp_re_ = /&/g;
var lt_re_ = /</g;
var gt_re_ = />/g;

// Convert text to HTML format. For efficiency, we just convert '&', '<', '>'
// characters.
// Note: Javascript >= 1.3 supports lambda expression in the replacement
// argument. But it's slower on IE.
// Note: we can also implement HtmlEscape by setting the value
// of a textnode and then reading the 'innerHTML' value, but that
// that turns out to be slower.
// Params: str: String to be escaped.
// Returns: The escaped string.
function HtmlEscape(str) {
  if (!str) return "";
  return str.replace(amp_re_, "&amp;").replace(lt_re_, "&lt;").
    replace(gt_re_, "&gt;").replace(quote_re_, "&quot;");
}

/** converts html entities to plain text.  It covers the most common named
 * entities and numeric entities.
 * It does not cover all named entities -- it covers &{lt,gt,amp,quot,nbsp}; but
 * does not handle some of the more obscure ones like &{ndash,eacute};.
 */
function HtmlUnescape(str) {
  if (!str) return "";
  return str.
    replace(/&#(\d+);/g,
      function (_, n) { return String.fromCharCode(parseInt(n, 10)); }).
    replace(/&#x([a-f0-9]+);/gi,
      function (_, n) { return String.fromCharCode(parseInt(n, 16)); }).
    replace(/&(\w+);/g, function (_, entity) {
      entity = entity.toLowerCase();
      return entity in HtmlUnescape.unesc ? HtmlUnescape.unesc[entity] : '?';
    });
}
HtmlUnescape.unesc = { lt: '<', gt: '>', quot: '"', nbsp: ' ', amp: '&' };

// Escape double quote '"' characters in addition to '&', '<', '>' so that a
// string can be included in an HTML tag attribute value within double quotes.
// Params: str: String to be escaped.
// Returns: The escaped string.
var quote_re_ = /\"/g;

var JS_SPECIAL_RE_ = /[\'\\\r\n\b\"<>&]/g;

function JSEscOne_(s) {
  if (!JSEscOne_.js_escs_) {
    var escapes = {};
    escapes['\\'] = '\\\\';
    escapes['\''] = '\\047';
    escapes['\n'] = '\\n';
    escapes['\r'] = '\\r';
    escapes['\b'] = '\\b';
    escapes['\"'] = '\\042';
    escapes['<'] =  '\\074';
    escapes['>'] =  '\\076';
    escapes['&'] =  '\\046';

    JSEscOne_.js_escs_ = escapes;
  }

  return JSEscOne_.js_escs_[s];
}

// converts multiple ws chars to a single space, and strips
// leading and trailing ws
var spc_re_ = /\s+/g;
var beg_spc_re_ = /^ /;
var end_spc_re_ = / $/;
function CollapseWhitespace(str) {
  if (!str) return "";
  return str.replace(spc_re_, " ").replace(beg_spc_re_, "").
    replace(end_spc_re_, "");
}

var newline_re_ = /\r?\n/g;
var spctab_re_ = /[ \t]+/g;
var nbsp_re_ = /\xa0/g;

function HtmlifyNewlines(str) {
  if (!str) return "";
  return str.replace(newline_re_, "<br>");
}

// URL encodes the string.
function UrlEncode(str) {
  return encodeURIComponent(str);
}

function Trim(str) {
  if (!str) return "";
  return str.replace(/^\s+/, "").replace(/\s+$/, "");
}

function EndsWith(str, suffix) {
  if (!str) return !suffix;
  return (str.lastIndexOf(suffix) == (str.length - suffix.length));
}

// Check if a string is empty
function IsEmpty(str) {
  return CollapseWhitespace(str) == "";
}

// Check if a character is a letter
function IsLetterOrDigit(ch) {
  return ((ch >= "a" && ch <= "z") ||
          (ch >= "A" && ch <= "Z") ||
         (ch >= '0' && ch <= '9'));
}

// Check if a character is a space character
function IsSpace(ch) {
  return (" \t\r\n".indexOf(ch) >= 0);
}

//------------------------------------------------------------------------
// TextArea utilities
//------------------------------------------------------------------------

function SetCursorPos(win, textfield, pos) {
  if (IsDefined(textfield.selectionEnd) &&
      IsDefined(textfield.selectionStart)) {
    // Mozilla directly supports this
    textfield.selectionStart = pos;
    textfield.selectionEnd = pos;

  } else if (win.document.selection && textfield.createTextRange) {
    // IE has textranges. A textfield's textrange encompasses the
    // entire textfield's text by default
    var sel = textfield.createTextRange();

    sel.collapse(true);
    sel.move("character", pos);
    sel.select();
  }
}

//------------------------------------------------------------------------
// Array utilities
//------------------------------------------------------------------------
// Find an item in an array, returns the key, or -1 if not found
function FindInArray(array, x) {
  for (var i = 0; i < array.length; i++) {
    if (array[i] == x) {
      return i;
    }
  }
  return -1;
}

// Inserts an item into an array, if it's not already in the array
function InsertArray(array, x) {
  if (FindInArray(array, x) == -1) {
    array[array.length] = x;
  }
}

// Delete an element from an array
function DeleteArrayElement(array, x) {
  var i = 0;
  while (i < array.length && array[i] != x)
    i++;
  array.splice(i, 1);
}

function GetEventTarget(/*Event*/ ev) {
// Event is not a type in IE; IE uses Object for events
//  AssertType(ev, Event, 'arg passed to GetEventTarget not an Event');
  return ev.srcElement || ev.target;
}

//------------------------------------------------------------------------
// Misc
//------------------------------------------------------------------------
// Check if a value is defined
function IsDefined(value) {
  return (typeof value) != 'undefined';
}

function GetKeyCode(event) {
  var code;
  if (event.keyCode) {
    code = event.keyCode;
  } else if (event.which) {
    code = event.which;
  }
  return code;
}

// define a forid function to fetch a DOM node by id.
function forid_1(id) {
  return document.getElementById(id);
}
function forid_2(id) {
  return document.all[id];
}

/**
 * Fetch an HtmlElement by id.
 * DEPRECATED: use $ in dom.js
 */
var forid = document.getElementById ? forid_1 : forid_2;



function log(msg) {
  /* a top level window is its own parent.  Use != or else fails on IE with
   * infinite loop.
   */
  try {
    if (window.parent != window && window.parent.log) {
      window.parent.log(window.name + '::' + msg);
      return;
    }
  } catch (e) {
    // Error: uncaught exception: Permission denied to get property Window.log
  }
  var logPane = forid('log');
  if (logPane) {
    var logText = '<p class=logentry><span class=logdate>' + new Date() +
                  '</span><span class=logmsg>' + msg + '</span></p>';
    logPane.innerHTML = logText + logPane.innerHTML;
  } else {
    window.status = msg;
  }
}
