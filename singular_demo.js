// Alfredo Canziani, 18 Aug 2021
/// <reference path="./node_modules/ml-matrix/matrix.d.ts" />
/// <reference path="./node_modules/@types/p5/global.d.ts" />

if (typeof(require) !== "undefined") {
    const mlMatrix = require("ml-matrix");
}

/*
 * @mmdts:
 * I split all "global" variables in this code file into one of three things:
 * 1) A global constant, like RED, GREEN, and BLUE_ORANGE. These are never assigned, only read from.
 * 2) A state variable, like A, svd, det, locked, and clicked. These are assigned and read from across
 *    multiple functions, therefore the global state. Ideally, a way to write this code exists in which
 *    those are all parts of a given class, and that class has code for managing ML, and interacts with
 *    another class that's responsible for the "drawing" aspect, sort of a separation-of-concerns thing.
 *    But this is not in the scope of the changes, since it would make the code look not like something
 *    it was, but rather like something completely new (but it would be the right direction to take).
 * 3) Methods/properties defined on window, which are put there by p5 (a very intrusive canvas library).
 *    The reason that p5 cannot be made into a module is because it messes with the window object so
 *    much, which was good practice in 2008 but is now very bad coding practice in 2021 given how the JS
 *    scene has evolved, but it's what they opted to do. So I made all of those little intrusions clear
 *    by adding a "window." at the stat of each of them to mark that those belong to the window object.
 *    This is advisable behavior in JS coding.
 *
 * Things I noticed while fiddling around with the code:
 * - You can drag the circle before setting both eigenvectors, and that allows the entire vector to deform,
 *   as shown in deform.png in this repository. If you select a second eigenvector after having dragged the
 *   circle, you no longer become able to drag anything again until you refresh the page! Even resetting
 *   does not fix this...
 * - Allowing circle highlighting by uncommenting the lines found by searching for @mmdts[1] sometimes
 *   causes subsequent clicks to not work as expected? I haven't tried this long enough to figure the
 *   reason out though.
 *
 * Keep in mind all calculation done in the JS files sourced by the served HTML file are all run in the
 * browser, and therefore client-side. Client-side Javascript (even with WASM support) cannot call CUDA C++
 * code, and can only run on single-processor-core performance, and is therefore very slow when it comes to
 * more complicated machine learning algorithms like networks with thousands of parameters!
 *
 * This is a shortcoming of the browser technology, which for safety reasons, do not allow you to invoke
 * external programs from the sandboxed browser context that browser JS runs in.
 *
 * Just wanted to make that perfectly clear in case it wasn't.
 *
 * If you want to have faster running JS simulations, consider using Python3 for machine learning, then
 * serving some HTML+JS page on the browser using flask or some simpler HTTP server to interact with the
 * python app, query algorithms, and show the results.
 *
 * This is how most web interfaces for deep learning projects work, and it's how prototype robotics that
 * offer a web interface and use computer vision work.
 */

// 2 bases, 2 left-singular vectors, 2 e'evectors

const RED = '#fb8072';
const GREEN = '#b3de69';
const BLUE_ORANGE = ['#80b1d3', '#fdb462'];

const NORM = 50;
const BASES = [0, 12];
const N = 16;

let state = {}

state.A = [];
state.svd = null;
state.det = null;
state.locked = [false, false, false, false, false, false]
state.s_values = [[NORM, 0], [0, -NORM]]

state.V = [null, null]
state.U = [null, null]
state.clicked = []
state.e_values = [1, 1]


function mult(a, b) {
    const ux = a[0][0] * b[0] + a[0][1] * b[1];
    const uy = a[1][0] * b[0] + a[1][1] * b[1];
    return [ux, uy];
}

function compute_singular_vectors(input_matrix) {
    const svd = new mlMatrix.SVD(input_matrix);

    // let S = mlMatrix.Matrix.diag(svd.s);
    // let R = svd.U.mmul(mlMatrix.Matrix.diag(svd.s)).mmul(svd.V.transpose());
    // let r = R.to1DArray().map(u => u.toFixed(2)).toString();
    // let a = input_matrix.toString();

    // Fucking determinant!
    const det = mlMatrix.determinant(svd.U);
    // console.log(det.toFixed(2))

    return [svd, det];
}

function update_matrix() {
    const [u1, u2] = state.V[state.clicked[0]];
    const [v1, v2] = state.V[state.clicked[1]];
    const [l1, l2] = state.e_values;

    const d = u1 * v2 - u2 * v1
    state.A[0][0] = (v2 * l1 * u1 - u2 * l2 * v1) / d
    state.A[0][1] = (u1 * l2 * v1 - v1 * l1 * u1) / d
    state.A[1][0] = (v2 * l1 * u2 - u2 * l2 * v2) / d
    state.A[1][1] = (u1 * l2 * v2 - v1 * l1 * u2) / d
}

window.setup = function () {
    window.createCanvas(
        Math.min(window.innerWidth, 600),
        Math.min(window.innerHeight, 400)
    );

    // Centre coordinates
    state.CX = window.width / 2;
    state.CY = window.height / 2;
    for (let n = 0; n < N; n++) {
        state.V[n] = [];
        state.V[n][0] = NORM * Math.cos(n * 2 * Math.PI / N);
        state.V[n][1] = NORM * Math.sin(n * 2 * Math.PI / N);
        state.U[n] = [...state.V[n]];
    }

    state.A[0] = [1, 0];
    state.A[1] = [0, 1];

    [state.svd, state.det] = compute_singular_vectors(state.A);

    // Create reset button
    let button = window.createButton('Reset');
    button.position(0, height - 20);
    button.mousePressed(reset);
    button.style('background-color', 'black');
    button.style('color', '#bbbbbb');
    button.style('border', 'none');
}


window.draw = function () {
    background("black")
    fill("#bbbbbb")

    if (state.clicked.length < 2)
      window.text("Select two eigenvectors", 20, 30)
    if (state.clicked.length === 2) {
      window.text("Drag 'em around", 20, 30)
      update_matrix()
    }

    for (let i = 0; i < 2; i++) {
        const sx = state.svd.U.data[i][0] * state.svd.s[i] * NORM + state.CX
        const sy = -state.svd.U.data[i][1] * state.svd.s[i] * NORM * state.det + state.CY
        const d = 10;
        window.fill(BLUE_ORANGE[i]);
        window.noStroke()
        window.ellipse(sx, sy, d)
        window.stroke(BLUE_ORANGE[i])
        window.line(sx, sy, 2 * state.CX - sx, 2 * state.CY - sy)
    }

    for (let n = 0; n < N; n++) {
        // Update u vectors
        state.U[n] = mult(state.A, state.V[n]);

        // Move u and v to the centre
        let ux = state.U[n][0] + state.CX
        let uy = state.U[n][1] + state.CY
        let vx = state.V[n][0] + state.CX
        let vy = state.V[n][1] + state.CY

        // Draw the input points in grey
        // Draw i and j in R and G
        window.fill('grey');
        switch (n) {
            case BASES[0]:
                fill(RED);
                break
            case BASES[1]:
                fill(GREEN);
                break
        }
        let d = 5;
        window.noStroke();
        window.ellipse(vx, vy, d);

        // Set output points in white
        window.fill('white');

        // Highlight circles under the mouse @mmdts[1]
        // if (dist(window.mouseX, window.mouseY, ux, uy) <= 5 && state.clicked.length < 2)
        //   d = 15;

        // Draw lines through the e'vectors, set colour and size
        for (let i = 0; i < state.clicked.length; i++)
            if (state.clicked[i] === n) {
                window.stroke("grey");
                window.line(
                    state.CX - state.V[n][0] * 10, state.CY - state.V[n][1] * 10,
                    state.CX + state.V[n][0] * 10, state.CY + state.V[n][1] * 10
                );
                d = 10;
                window.fill(181, 223, 108);
            }

        // Draw i and j in R and G
        switch (n) {
            case BASES[0]:
                d = 10;
                window.fill(RED);
                break;
            case BASES[1]:
                d = 10;
                window.fill(GREEN);
                break;
        }


        // Draw output points
        window.ellipse(ux, uy, d);
    }
}

window.mousePressed = function () {

    // Selecting e'vectors
    if (state.clicked.length < 2)
      for (let n = 0; n < N; n++) {
        const vx = state.V[n][0] + state.CX
        const vy = state.V[n][1] + state.CY
        if (dist(window.mouseX, window.mouseY, vx, vy) <= 5 && !state.clicked.includes(n))
          state.clicked.push(n);
      }

    // Holding on e'vector
    if (state.clicked.length === 2)
        for (let i = 0; i < 2; i++) {
            const ux = state.U[state.clicked[i]][0] + state.CX;
            const uy = state.U[state.clicked[i]][1] + state.CY;
            if (dist(window.mouseX, window.mouseY, ux, uy) <= 5)
                state.locked[i] = true;
        }

    // Holding on bases vectors
    for (let i = 0; i < 2; i++) {
        const ux = state.U[BASES[i]][0] + state.CX;
        const uy = state.U[BASES[i]][1] + state.CY;
        if (dist(window.mouseX, window.mouseY, ux, uy) <= 5)
            state.locked[i] = true;
    }

}

window.mouseReleased = function () {
    for (let i = 0; i < 6; i++)
        state.locked[i] = false;

}

window.mouseDragged = function () {
    // for (let i = 0; i < 2; i++) {  // Check both e'vectors
    //     if (state.locked[i]) {           // If I'm clicking on it
    //         let d;
    //         // state.V[state.clicked[i]][0] = window.mouseX - state.CX
    //         // state.V[state.clicked[i]][1] = window.mouseY - state.CY
    //         // d = dist(window.mouseX, window.mouseY, state.CX, state.CY) / NORM
    //         d = ((window.mouseX - state.CX) * state.V[state.clicked[i]][0] +
    //             (window.mouseY - state.CY) * state.V[state.clicked[i]][1]) / NORM ** 2
    //         state.e_values[i] = d
    //         // state.A[i][i] = d
    //     }
    // }
    for (let i = 0; i < 2; i++)
        if (state.locked[i]) {
            state.A[0][i] = (window.mouseX - state.CX) / NORM * (-1) ** i;
            state.A[1][i] = (window.mouseY - state.CY) / NORM * (-1) ** i;
            [state.svd, state.det] = compute_singular_vectors(state.A);
        }
}

function reset() {
    state.A[0] = [1, 0]
    state.A[1] = [0, 1]
    state.clicked = []
    state.e_values = [1, 1]
    state.locked = [false, false]
}